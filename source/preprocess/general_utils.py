import torch_geometric
import torch
#import torch_scatter
import math
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
import rdkit
import rdkit.Chem
import rdkit.Chem.AllChem
import rdkit.Chem.rdMolTransforms
import networkx as nx
import random
from tqdm import tqdm
from rdkit.Chem import rdMolTransforms
from rdkit.Geometry import Point3D

import pdb

atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I']
formalCharge = [-1, -2, 1, 2, 0]
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
num_single_bonds = [0,1,2,3,4,5,6]
num_double_bonds = [0,1,2,3,4]
num_triple_bonds = [0,1,2]
num_aromatic_bonds = [0,1,2,3,4]
bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding

def getNodeFeatures(list_rdkit_atoms):
    F_v = (len(atomTypes)+1)
    F_v += (len(formalCharge)+1)
    F_v += (1 + 1)
    
    F_v += 8
    F_v += 6
    F_v += 4
    F_v += 6
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes) # atom symbol, dim=12 + 1 
        features += one_hot_embedding(node.GetFormalCharge(), formalCharge) # formal charge, dim=5+1 
        features += [int(node.GetIsAromatic())] # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass()  * 0.01] # atomic mass / 100, dim=1
        
        atom_bonds = np.array([b.GetBondTypeAsDouble() for b in node.GetBonds()])
        N_single = int(sum(atom_bonds == 1.0) + node.GetNumImplicitHs() + node.GetNumExplicitHs())
        N_double = int(sum(atom_bonds == 2.0))
        N_triple = int(sum(atom_bonds == 3.0))
        N_aromatic = int(sum(atom_bonds == 1.5))
        
        features += one_hot_embedding(N_single, num_single_bonds)
        features += one_hot_embedding(N_double, num_double_bonds)
        features += one_hot_embedding(N_triple, num_triple_bonds)
        features += one_hot_embedding(N_aromatic, num_aromatic_bonds)
        
        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)

def get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment):
    bonds_indices = [b.GetIdx() for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds() if len(set([b.GetBeginAtomIdx(), b.GetEndAtomIdx()]).intersection(set(ring_fragment))) == 1]
    bonded_atom_indices_sorted = [(b[0], b[1]) if (b[0] in ring_fragment) else (b[1], b[0]) for b in bonded_atom_indices]
    atoms = [b[1] for b in bonded_atom_indices_sorted]
    
    return bonds_indices, bonded_atom_indices_sorted, atoms

def get_fragment_smiles(mol, ring_fragment):
    ring_fragment = [int(r) for r in ring_fragment]
    
    bonds_indices, bonded_atom_indices_sorted, atoms_bonded_to_ring = get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment)
    
    pieces = rdkit.Chem.FragmentOnSomeBonds(mol, bonds_indices, numToBreak=len(bonds_indices), addDummies=False) 

    fragsMolAtomMapping = []
    fragments = rdkit.Chem.GetMolFrags(pieces[0], asMols = True, sanitizeFrags = True, fragsMolAtomMapping = fragsMolAtomMapping)
    
    frag_mol = [m_ for i,m_ in enumerate(fragments) if (set(fragsMolAtomMapping[i]) == set(ring_fragment))][0]
    
    for a in range(frag_mol.GetNumAtoms()):
        N_rads = frag_mol.GetAtomWithIdx(a).GetNumRadicalElectrons()
        N_Hs = frag_mol.GetAtomWithIdx(a).GetTotalNumHs()
        if N_rads > 0:
            frag_mol.GetAtomWithIdx(a).SetNumExplicitHs(N_rads + N_Hs)
            frag_mol.GetAtomWithIdx(a).SetNumRadicalElectrons(0)
    
    smiles = rdkit.Chem.MolToSmiles(frag_mol, isomericSmiles = False)
    
    smiles_mol = rdkit.Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        logger(f'failed to extract fragment smiles: {smiles}, {ring_fragment}')

        return None

    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    
    return reduced_smiles

def get_multiple_bonds_to_ring(mol):
    BondToRing = rdkit.Chem.MolFromSmarts('[r]!@[*]')
    bonds_to_rings = mol.GetSubstructMatches(BondToRing)
    NonSingleBond = rdkit.Chem.MolFromSmarts('[*]!-[*]')
    non_single_bonds = mol.GetSubstructMatches(NonSingleBond)
    
    bonds_to_rings = [tuple(sorted(b)) for b in bonds_to_rings]
    non_single_bonds = [tuple(sorted(b)) for b in non_single_bonds]
    
    return tuple(set(bonds_to_rings).intersection(set(non_single_bonds)))

def get_rigid_ring_linkers(mol):
    RingLinker = rdkit.Chem.MolFromSmarts('[r]!@[r]')
    ring_linkers = mol.GetSubstructMatches(RingLinker)
    
    NonSingleBond = rdkit.Chem.MolFromSmarts('[*]!-[*]')
    non_single_bonds = mol.GetSubstructMatches(NonSingleBond)
    
    ring_linkers = [tuple(sorted(b)) for b in ring_linkers]
    non_single_bonds = [tuple(sorted(b)) for b in non_single_bonds]
    
    return tuple(set(ring_linkers).intersection(set(non_single_bonds)))

def get_rings(mol):
    return mol.GetRingInfo().AtomRings()

def get_ring_fragments(mol):
    rings = get_rings(mol)
    
    rings = [set(r) for r in rings]
    
    # combining rigid ring structures connected by rigid (non-single) bond (they will be combined in the next step)
    rigid_ring_linkers = get_rigid_ring_linkers(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in rigid_ring_linkers:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings
    
    # joining ring structures
    N_rings = len(rings)
    done = False
    
    joined_rings = []
    for i in range(0, len(rings)):
        
        joined_ring_i = set(rings[i])            
        done = False
        while not done:
            for j in range(0, len(rings)): #i+1
                ring_j = set(rings[j])
                if (len(joined_ring_i.intersection(ring_j)) > 0) & (joined_ring_i.union(ring_j) != joined_ring_i):
                    joined_ring_i = joined_ring_i.union(ring_j)
                    done = False
                    break
            else:
                done = True
        
        if joined_ring_i not in joined_rings:
            joined_rings.append(joined_ring_i)
    
    rings = joined_rings
    
    # adding in rigid (non-single) bonds to these ring structures
    multiple_bonds_to_rings = get_multiple_bonds_to_ring(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in multiple_bonds_to_rings:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings
    
    return rings

def generate_conformer(smiles, addHs = False):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    mol = rdkit.Chem.AddHs(mol)
    rdkit.Chem.AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    
    if not addHs:
        mol = rdkit.Chem.RemoveHs(mol)
    return mol


def get_ring_fragments(mol):
    rings = get_rings(mol)
    
    rings = [set(r) for r in rings]
    
    # combining rigid ring structures connected by rigid (non-single) bond (they will be combined in the next step)
    rigid_ring_linkers = get_rigid_ring_linkers(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in rigid_ring_linkers:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings
    
    # joining ring structures
    N_rings = len(rings)
    done = False
    
    joined_rings = []
    for i in range(0, len(rings)):
        
        joined_ring_i = set(rings[i])            
        done = False
        while not done:
            for j in range(0, len(rings)): #i+1
                ring_j = set(rings[j])
                if (len(joined_ring_i.intersection(ring_j)) > 0) & (joined_ring_i.union(ring_j) != joined_ring_i):
                    joined_ring_i = joined_ring_i.union(ring_j)
                    done = False
                    break
            else:
                done = True
        
        if joined_ring_i not in joined_rings:
            joined_rings.append(joined_ring_i)
    
    rings = joined_rings
    
    # adding in rigid (non-single) bonds to these ring structures
    multiple_bonds_to_rings = get_multiple_bonds_to_ring(mol)
    new_rings = []
    for ring in rings:
        new_ring = ring
        for bond in multiple_bonds_to_rings:
            if (bond[0] in ring) or (bond[1] in ring):
                new_ring = new_ring.union(set(bond))
        new_rings.append(new_ring)
    rings = new_rings
    
    return rings

def get_fragment_smiles(mol, ring_fragment):
    ring_fragment = [int(r) for r in ring_fragment]
    
    bonds_indices, bonded_atom_indices_sorted, atoms_bonded_to_ring = get_singly_bonded_atoms_to_ring_fragment(mol, ring_fragment)
    
    if len(bonds_indices) == 0:
        return rdkit.Chem.MolToSmiles(mol)
    
    pieces = rdkit.Chem.FragmentOnSomeBonds(mol, bonds_indices, numToBreak=len(bonds_indices), addDummies=False)

    fragsMolAtomMapping = []
    fragments = rdkit.Chem.GetMolFrags(pieces[0], asMols = True, sanitizeFrags = True, fragsMolAtomMapping = fragsMolAtomMapping)
    
    frag_mol = [m_ for i,m_ in enumerate(fragments) if (set(fragsMolAtomMapping[i]) == set(ring_fragment))][0]
    
    for a in range(frag_mol.GetNumAtoms()):
        N_rads = frag_mol.GetAtomWithIdx(a).GetNumRadicalElectrons()
        N_Hs = frag_mol.GetAtomWithIdx(a).GetTotalNumHs()
        if N_rads > 0:
            frag_mol.GetAtomWithIdx(a).SetNumExplicitHs(N_rads + N_Hs)
            frag_mol.GetAtomWithIdx(a).SetNumRadicalElectrons(0)
    
    smiles = rdkit.Chem.MolToSmiles(frag_mol, isomericSmiles = False)
    
    smiles_mol = rdkit.Chem.MolFromSmiles(smiles)
    if not smiles_mol:
        return None

    reduced_smiles = rdkit.Chem.MolToSmiles(smiles_mol, isomericSmiles = False)
    
    return reduced_smiles