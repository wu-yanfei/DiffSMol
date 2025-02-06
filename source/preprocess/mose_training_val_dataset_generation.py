import torch_geometric
import torch
import torch_scatter
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import rdkit
from rdkit import Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import rdkit.Chem.rdmolops
from rdkit.Chem.rdmolops import AssignStereochemistryFrom3D
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

import networkx as nx
import random
from tqdm import tqdm
import pickle
import os, sys
import pdb
from functools import partial
from .generate_artificial_mols_MOSES2 import fix_bonding_geometries

import collections
import collections.abc
from .general_utils import *

from multiprocessing import Pool

def get_acyclic_single_bonds(mol):
    AcyclicBonds = rdkit.Chem.MolFromSmarts('[*]!@[*]')
    SingleBonds = rdkit.Chem.MolFromSmarts('[*]-[*]')
    acyclicBonds = mol.GetSubstructMatches(AcyclicBonds)
    singleBonds = mol.GetSubstructMatches(SingleBonds)
    
    acyclicBonds = [tuple(sorted(b)) for b in acyclicBonds]
    singleBonds = [tuple(sorted(b)) for b in singleBonds]
    
    select_bonds = set(acyclicBonds).intersection(set(singleBonds))
    return select_bonds

def flatten(x):
    if isinstance(x, collections.abc.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def logger(text, file = '../data/MOSES2/MOSES2_training_val_terminalSeeds_generation_reduced_log.txt'):
    with open(file, 'w') as f:
        f.write(text + '\n')
        
        
def conformer_generation(s):
    # s doesn't have stereochemistry specified
    try:
        m_ = rdkit.Chem.MolFromSmiles(s) 
        s_nostereo = rdkit.Chem.MolToSmiles(m_, isomericSmiles = False)
        
        # generate MMFF-optimized conformer and assign stereochemistry from 3D
        m = rdkit.Chem.MolFromSmiles(s_nostereo)
        m = rdkit.Chem.AddHs(m)
        rdkit.Chem.AllChem.EmbedMolecule(m)
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(m, maxIters = 200)
        m = rdkit.Chem.RemoveHs(m)
        m.GetConformer()
        
        n = len(rdkit.Chem.rdmolops.GetMolFrags(m))
        assert n == 1
        
        AssignStereochemistryFrom3D(m)
        
    except:
        m = None
    
    return m

def is_filtered(mol, top_100_fragments_smiles):
    is_filtered = False
    ring_fragments = get_ring_fragments(mol)
    for frag in ring_fragments:
        smiles = get_fragment_smiles(mol, frag)
        if smiles not in top_100_fragments_smiles:
            break
    else:
        is_filtered = True
    return is_filtered

if __name__ == "__main__":
    logger('creating MOSES2 mol database...')
        
    smiles_df = pd.read_csv('../data/MOSES2/train_MOSES.csv')
    smiles = list(smiles_df.SMILES)

    mols = []

    pool = Pool()    
    for i, m in tqdm(enumerate(pool.imap(conformer_generation, smiles)), total = len(smiles)):
        if m is not None:
            mols.append(m)
    pool.close()
    pool.join()

    rdkit_smiles = [rdkit.Chem.MolToSmiles(m) for m in mols]
    rdkit_smiles_nostereo = [rdkit.Chem.MolToSmiles(m, isomericSmiles = False) for m in mols]

    database = pd.DataFrame()
    database['ID'] = rdkit_smiles
    database['SMILES_nostereo'] = rdkit_smiles_nostereo
    database['rdkit_mol_cistrans_stereo'] = mols
    database['N_atoms'] = [m.GetNumAtoms() for m in mols]
    database['N_rot_bonds'] = [len(get_acyclic_single_bonds(m)) for m in mols]

    database = database.drop_duplicates('ID').reset_index(drop = True)

    logger(f'database has {len(database)} entries')

    bad_confs = []
    for m, mol_db in enumerate(database.rdkit_mol_cistrans_stereo):
        try:
            mol_db.GetConformer()
            bad_confs.append(1)
        except:
            bad_confs.append(0)

    database['has_conf'] = bad_confs
    database = database[database.has_conf == 1].reset_index(drop = True)

    # =================== extract top 100 fragments ===================

    atom_fragments = pd.read_pickle("../data/MOSES2/MOSES2_training_val_AtomFragment_database.pkl")['mol']
    top_100_fragments_smiles = [Chem.MolToSmiles(tmp) for tmp in atom_fragments if tmp is not None]

    # =================== add filter identifier to signify molecules with fragments not covered by SQUID ===========

    filtered_database_SMILES_nostereo = []
    m = 0
    SMILES_nostereo_reduced_dataset = database.drop_duplicates('SMILES_nostereo')

    total = len(SMILES_nostereo_reduced_dataset)
    is_filtered_list = []
    num = 100

    unfiltered_mols = SMILES_nostereo_reduced_dataset.rdkit_mol_cistrans_stereo
    if_filtered_masks = []

    for i in range(num):
        logger(str(i))
        
        mols = deepcopy(unfiltered_mols[i*int((len(unfiltered_mols) / float(num)) + 1.) : (i+1)*int((len(unfiltered_mols) / float(num)) + 1.)])
        logger(f'{len(mols)} mols in subset')
        
        filtered_mask = []
        func = partial(is_filtered, top_100_fragments_smiles=top_100_fragments_smiles)
        pool = Pool(20)
        for inc, if_filter in tqdm(enumerate(pool.imap(func, mols)), total = len(mols)):
            filtered_mask.append(if_filter)
            if inc in [int((total / 20) * n) for n in range(1, 21)]:
                logger(f'    {(inc / total) * 100.} % complete...')
        
        if_filtered_masks.extend(filtered_mask)

    SMILES_nostereo_reduced_dataset['original_index'] = list(range(0, len(SMILES_nostereo_reduced_dataset)))
    SMILES_nostereo_reduced_dataset['is_filtered'] = if_filtered_masks

    # ============================= add artificial molecules ===========================

    bond_lookup = pd.read_pickle('../data/MOSES2/MOSES2_training_val_bond_lookup.pkl')
    unique_atoms = np.load('../data/MOSES2/MOSES2_training_val_unique_atoms.npy')

    all_mols = list(SMILES_nostereo_reduced_dataset.rdkit_mol_cistrans_stereo)
        
    all_artificial_mols = []
        
    logger(f'add artificial mols for {len(all_mols)} total mols ')

    num = 100
    for i in range(num):
        logger(str(i))
        
        mols = deepcopy(all_mols[i*int((len(all_mols) / float(num)) + 1.) : (i+1)*int((len(all_mols) / float(num)) + 1.)])
        logger(f'{len(mols)} mols in subset')
        if len(mols) == 0: continue
        artificial_mols = []
        fails = 0
        
        pool = Pool(20)    
        for i, tup in tqdm(enumerate(pool.imap(fix_bonding_geometries, mols)), total = len(mols)):
            m_a, fail = tup
            artificial_mols.append(m_a)
            fails += fail
        pool.close()
        pool.join()
        
        logger(f'{fails} total fails ... {fails/len(mols)}% failure rate.')

        all_artificial_mols += artificial_mols
        
    print(f'{len(all_artificial_mols)} total artificial mols')
        
    SMILES_nostereo_reduced_dataset['artificial_mol'] = all_artificial_mols

    logger('saving filtered database with artificial molecules...')
    SMILES_nostereo_reduced_dataset.to_pickle('../data/MOSES2/MOSES2_training_val_dataset.pkl')

    # ====================== train / val split =======================================


    logger('creating training/val splits...')
    all_smiles = list(set(list(SMILES_nostereo_reduced_dataset.SMILES_nostereo)))
    random.shuffle(all_smiles)

    train_smiles = all_smiles[0:int(len(all_smiles)*0.8)]
    val_smiles = all_smiles[int(len(all_smiles)*0.8):]

    train_smiles_df = pd.DataFrame()
    train_smiles_df['SMILES_nostereo'] = train_smiles
    train_smiles_df['N_atoms'] = [rdkit.Chem.MolFromSmiles(s).GetNumAtoms() for s in train_smiles]
    train_smiles_df['N_acyclic_single_bonds'] = [len(get_acyclic_single_bonds(rdkit.Chem.MolFromSmiles(s))) for s in train_smiles]

    val_smiles_df = pd.DataFrame()
    val_smiles_df['SMILES_nostereo'] = val_smiles
    val_smiles_df['N_atoms'] = [rdkit.Chem.MolFromSmiles(s).GetNumAtoms() for s in val_smiles]
    val_smiles_df['N_acyclic_single_bonds'] = [len(get_acyclic_single_bonds(rdkit.Chem.MolFromSmiles(s))) for s in val_smiles]

    train_smiles_df.to_csv('../data/MOSES2/MOSES2_train_smiles_split.csv')
    val_smiles_df.to_csv('../data/MOSES2/MOSES2_val_smiles_split.csv')
