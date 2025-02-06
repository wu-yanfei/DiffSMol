import argparse
import os
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter
from torch_scatter import scatter_sum, scatter_mean

from utils import eval_bond_length
from utils import eval_atom_type
from utils import analyze
from utils import misc
from utils import scoring_func
from utils import reconstruct
from utils import transforms
from utils.docking import QVinaDockingTask
import multiprocessing as mp


def evaluate_pocket(result_fn):
    ligand_results = torch.load(result_fn) #os.path.join(args.sample_path, f'result_{ligand_id}.pt'))
    ligand_id = int(os.path.basename(result_fn)[:-3].split('_')[-1])
    all_pred_ligand_pos = ligand_results['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
    all_pred_ligand_v = ligand_results['pred_ligand_v_traj']
    num_samples = len(all_pred_ligand_pos)

    eval_results = []
    for sample_idx, (pred_pos, pred_v) in enumerate(
            tqdm(zip(all_pred_ligand_pos, all_pred_ligand_v), desc='Eval', total=num_samples)):
        pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
        if args.pos_offset:
            #protein_pos = pocket_results['data'].protein_pos
            #center_pos = torch.mean(protein_pos, dim=0, keepdim=True)
            center = ligand_results['data'].point_cloud_center
            pred_pos = pred_pos + center.numpy()

        pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
        # reconstruction
        try:
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic,
                                                         basic_mode=args.basic_mode)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            if args.verbose:
                logger.warning('Reconstruct failed %s' % f'{sample_idx}')
            continue

        if '.' in smiles:
            continue
        
        # chemical and docking check
        try:
            chem_results = scoring_func.get_chem(mol)
            if args.docking:
                vina_task = QVinaDockingTask.from_generated_mol(
                    mol, ligand_results['data'].ligand_filename, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()
            else:
                vina_results = None
        except:
            if args.verbose:
                logger.warning('Evaluation failed for %s' % f'{sample_idx}')
            continue

        eval_results.append({
            'mol': mol,
            'smiles': smiles,
            'ligand_filename': ligand_results['data'].ligand_filename,
            'pred_pos': pred_pos,
            'pred_v': pred_v,
            'chem_results': chem_results,
            'vina': vina_results
        })
    logger.info(f'Evaluate No. {ligand_id} done! {num_samples} samples in total. {len(eval_results)} eval success!')
    torch.save(eval_results, os.path.join(result_path, f'eval_{ligand_id}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    # parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--basic_mode', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking', type=eval, default=True)
    parser.add_argument('--pos_offset', type=eval, default=False)
    parser.add_argument('-n', '--num_processes', type=int, default=10)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'docking_eval_results_v2')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate')
    logger.info(args)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    #if args.eval_num_examples is not None:
    #    results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)

    # all_samples = []
    # for data_id in range(num_examples):
    #     # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
    #     s = torch.load(os.path.join(args.sample_path, f'result_{data_id}.pt'))
    #     all_samples.append(s)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    #with mp.Pool(args.num_processes) as p:
    #    # docked_samples = p.starmap(evaluate_pocket, zip(all_samples, list(range(num_examples))))
    for result_fn in results_fn_list:
        evaluate_pocket(result_fn)
        #docked_samples = p.map(evaluate_pocket, results_fn_list)#list(range(num_examples)))
