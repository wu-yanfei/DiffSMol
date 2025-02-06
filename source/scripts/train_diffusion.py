import os
import shutil
import argparse
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch.utils.data import Subset

from datasets import get_dataset
from datasets.shape_mol_dataset import ShapeMolDataset
from datasets.shape_mol_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
import utils.transforms as trans
import utils.misc as misc
import utils.train as utils_train
import utils.data as utils_data
from utils import analyze
from utils.reconstruct import reconstruct_from_generated, MolReconsError
from utils.eval_bond_length import get_pair_length_profile, eval_pair_length_profile, \
    plot_distance_hist, pair_distance_from_pos_v
from rdkit import Chem
import time
from sklearn.metrics import roc_auc_score
import pickle
from scripts.sample_diffusion_no_pocket import sample_diffusion_ligand
from utils.evaluation import compare_with_ref, Local3D
from utils import reconstruct
import pandas as pd
from utils import scoring_func
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

data_metrics = pd.read_csv("../data/MOSES2/metrics/test_metric_50k_mols2.csv")
with open("../data/MOSES2/metrics/test_metric_50k_mols2_local3d.pkl", 'rb') as f:
    data_local3d_dict = pickle.load(f)

def get_jsds_with_data(values, data):
    metric_list = list(values.keys())
    all_jsds = {}
    for metric in metric_list:
        value_list = values[metric]
        ref_list = data[metric]
        jsd = compare_with_ref(value_list, ref_list)
        if jsd is None or np.isnan(jsd): continue
        all_jsds[metric] = jsd
    return all_jsds

def validate_via_sample(model, batch_data, logger=None, num_samples=50, atom_enc_mode='add_aromatic'):
    n_recon_success, n_complete = 0, 0
    mols = []
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    all_sample_nums = 0
    all_chem_results = []
    for data in batch_data:
        all_ligand_pos, all_ligand_v, all_ligand_bond, _, _, _, _, _, _, _ = sample_diffusion_ligand(model, data, num_samples, batch_size=num_samples, sample_num_atoms='ref')
        #all_ligand_pos, all_ligand_v, all_ligand_bond = r['pos'], r['v'], r['bond']
        all_sample_nums += len(all_ligand_pos)
        
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_ligand_pos, all_ligand_v)):
            pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]

            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
            
            #if not model.pred_bond_type:
            try:
                mol_cal_bond = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                smiles = Chem.MolToSmiles(mol_cal_bond)
                mols.append(mol_cal_bond)
            except:
                continue
            #else:
            try:
                chem_results = scoring_func.get_chem(mol_cal_bond)
                all_chem_results.append(chem_results)
            except:
                continue

            #    try:
            #        pred_bond = all_ligand_bond[sample_idx]
            #        mol_pred_bond = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, bond=pred_bond)
            #        smiles = Chem.MolToSmiles(mol_pred_bond)
            #        mols.append(mol_pred_bond)
            #    except:
            #        continue
            
            n_recon_success += 1

            if "." not in smiles:
                n_complete += 1

    mean_qed = np.mean([results['qed'] for results in all_chem_results]) if len(all_chem_results) > 0 else 0
    mean_sa = np.mean([results['sa'] for results in all_chem_results]) if len(all_chem_results) > 0 else 0
    fraction_mol_stable = all_mol_stable / all_sample_nums
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / all_sample_nums
    fraction_complete = n_complete / all_sample_nums

    if len(mols) > 0:
        local3d = Local3D()
        local3d.get_predefined()
        #logger.info(f'Computing local 3d - bond lengths metric...')
        lengths = local3d.calc_frequent(mols, type_='length', parallel=False)
        #logger.info(f'Computing local 3d - bond angles metric...')
        angles = local3d.calc_frequent(mols, type_='angle', parallel=False)
        #logger.info(f'Computing local 3d - dihedral angles metric...')
        dihedrals = local3d.calc_frequent(mols, type_='dihedral', parallel=False)
        
        length_jsds = get_jsds_with_data(lengths, data_local3d_dict['lengths'])
        angle_jsds = get_jsds_with_data(angles, data_local3d_dict['angles'])
        dihedral_jsds = get_jsds_with_data(dihedrals, data_local3d_dict['dihedral'])

        mean_length_jsd = np.mean(list(length_jsds.values()))
        mean_angle_jsd = np.mean(list(angle_jsds.values()))
        mean_dihedral_jsd = np.mean(list(dihedral_jsds.values()))
        string = "length_jsd: \n"
        for metric in length_jsds:
            string += "%s %.6f\n" % (metric, length_jsds[metric])
        
        string += "angle_jsd: \n"
        for metric in angle_jsds:
            string += "%s %.6f\n" % (metric, angle_jsds[metric])

        string += "dihedral_jsd: \n"
        for metric in dihedral_jsds:
            string += "%s %.6f\n" % (metric, dihedral_jsds[metric])
        logger.info(string)
    else:
        mean_length_jsd, mean_angle_jsd, mean_dihedral_jsd = 100, 100, 100

    results = {
        'recon_success': fraction_recon,
        'complete': fraction_complete,
        'mol_stability': fraction_mol_stable, 
        'atom_stability': fraction_atm_stable,
        'mean_length_jsd': mean_length_jsd, 
        'mean_angle_jsd': mean_angle_jsd, 
        'mean_dihedral_jsd': mean_dihedral_jsd,
        'mean_qed': mean_qed,
        'mean_sa': mean_sa
    }

    out_string = "[Sample Validate] "
    for key in results:
        out_string += "%s: %.6f | " % (key, results[key])
        
    logger.info(out_string)

def get_auroc(y_true, y_pred, feat_mode=None, pred_type='atom'):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        if pred_type == 'atom':
            mapping = {
                'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
                'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
                'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL,
            }
            logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
        elif pred_type == 'bond':
            mapping = trans.MAP_INDEX_TO_BOND_TYPE
            logger.info(f'bond: {mapping[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_bond_auroc(y_true, y_pred):
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        bond_type = {
            0: 'none',
            1: 'single',
            2: 'double',
            3: 'triple',
            4: 'aromatic',
        }
        logger.info(f'bond: {bond_type[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='../logs_diffusion_full')
    parser.add_argument('--change_log_dir', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--continue_train_iter', type=int, default=-1)
    parser.add_argument('--train_report_iter', type=int, default=200)
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    if args.change_log_dir is not None:
        log_dir = args.change_log_dir
    else:
        log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    
    if args.change_log_dir is None:
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set = subsets['train'], subsets['valid']
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')
    
    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    model = ScorePosNet3D(
        config.model,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ligand_bond_feature_dim=len(utils_data.BOND_TYPES)
    ).to(args.device)
    
    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)
    
    start_iter = 1
    if args.continue_train_iter > 0 and os.path.exists(f'{ckpt_dir}/{args.continue_train_iter}.pt'):
        ckpt = torch.load(f'{ckpt_dir}/{args.continue_train_iter}.pt', map_location=args.device)
        model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
        logger.info(f'Successfully load the model! {args.continue_train_iter}.pt')
        start_iter = args.continue_train_iter + 1
        ckpt['optimizer']['param_groups'][-1]['lr'] = 1e-4
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    print(f'ligand feature dim: {ligand_featurizer.feature_dim}')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):
            batch = next(train_iterator).to(args.device)
            
            #t1 = time.time()
            results = model.get_diffusion_loss(
                    ligand_pos=batch.ligand_pos.float(),
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                    ligand_bond_index=batch.ligand_bond_index,
                    ligand_bond_type=batch.ligand_bond_type,
                    ligand_shape=batch.shape_emb if config.data.shape.use_shape else None,
                    eval_mode=False
            )
            
            loss, loss_pos, loss_v, loss_bond_final, loss_bond_aux = \
                results['loss'], results['loss_pos'], results['loss_v'], results['loss_bond_final'], results['loss_bond_aux']
            loss_bond_dist, loss_bond_angle, loss_torsion_angle = \
                results['loss_bond_dist'], results['loss_bond_angle'], results['loss_torsion_angle']
            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if it % args.train_report_iter == 0:
            logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | bond final %.6f | bond aux %.6f | bond_dist %.6f | bond_angle %.6f | torsion_angle %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                    it, loss, loss_pos, loss_v, loss_bond_final, loss_bond_aux, loss_bond_dist, loss_bond_angle, loss_torsion_angle, optimizer.param_groups[0]['lr'], orig_grad_norm
                )
            )
            for k, v in results.items():
                if torch.is_tensor(v) and v.squeeze().ndim == 0:
                    writer.add_scalar(f'train/{k}', v, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()

    def validate(it, sample_validate=True):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_loss_bond_final, sum_loss_bond_aux, sum_n = 0, 0, 0, 0, 0, 0
        sum_loss_bond_dist, sum_loss_bond_angle, sum_loss_torsion_angle = 0, 0, 0
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        
        inter_results = {}
        #if sample_validate:
        #    num_sample_validate = 64
        #else:
        #    num_sample_validate = 0
        # all_t_loss, all_t_loss_pos, all_t_loss_v = [], [], []
        with torch.no_grad():
            model.eval()
            #inter_results_path = os.path.join(ckpt_dir, 'inter_results%d.pt' % it)
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []

                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    
                    results = model.get_diffusion_loss(
                        ligand_pos=batch.ligand_pos.float(),
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        ligand_bond_index=batch.ligand_bond_index,
                        ligand_bond_type=batch.ligand_bond_type,
                        ligand_shape=batch.shape_emb,
                        eval_mode=True,
                        time_step=time_step
                    )
                    # loss, loss_pos, loss_v = results['vb_loss'], results['kl_pos'], results['kl_v']
                    loss, loss_pos, loss_v, loss_bond_final, loss_bond_aux = \
                        results['loss'], results['loss_pos'], results['loss_v'], results['loss_bond_final'], results['loss_bond_aux']
                    
                    loss_bond_dist, loss_bond_angle, loss_torsion_angle = \
                        results['loss_bond_dist'], results['loss_bond_angle'], results['loss_torsion_angle']
                    
                    sum_loss += float(loss) * batch_size
                    sum_loss_pos += float(loss_pos) * batch_size
                    sum_loss_v += float(loss_v) * batch_size
                    sum_loss_bond_final += float(loss_bond_final) * batch_size
                    sum_loss_bond_aux += float(loss_bond_aux) * batch_size
                    sum_loss_bond_dist += float(loss_bond_dist) * batch_size
                    sum_loss_bond_angle += float(loss_bond_angle) * batch_size
                    sum_loss_torsion_angle += float(loss_torsion_angle) * batch_size
                    sum_n += batch_size
                    
                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                    
                    if len(results['pred_bond_type']) != 0:
                        all_pred_bond_type.append(results['pred_bond_type'].detach().cpu().numpy())
                        all_gt_bond_type.append(results['gt_bond_type'].detach().cpu().numpy())
                
        
        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_loss_bond_final = sum_loss_bond_final / sum_n
        avg_loss_bond_aux = sum_loss_bond_aux / sum_n
        avg_loss_bond_dist = sum_loss_bond_dist / sum_n
        avg_loss_bond_angle = sum_loss_bond_angle / sum_n
        avg_loss_torsion_angle = sum_loss_torsion_angle / sum_n
        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)
        
        if len(all_pred_bond_type) != 0:
            bond_auroc = get_auroc(np.concatenate(all_gt_bond_type), np.concatenate(all_pred_bond_type, axis=0),
                                feat_mode=None, pred_type='bond')
        else:
            bond_auroc = 0.0
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(
            '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss bond final %.6f | Loss bond aux %.6f | '
            'Loss bond_dist %.6f | Loss bond_angle %.6f | Loss torsion_angle %.6f | Avg atom auroc %.6f | Avg bond auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_bond_final, avg_loss_bond_aux, avg_loss_bond_dist, avg_loss_bond_angle, avg_loss_torsion_angle, atom_auroc, bond_auroc
            )
        )

        if sample_validate and it > 50000 and it % 10000 == 0:
            sample_valid_data = [val_set[idx] for idx in np.random.choice(len(val_set), 2)]
            #try:
            results = validate_via_sample(model, sample_valid_data, logger)
            #except Exception as e:
            #    print(e)

        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/loss_pos', avg_loss_pos, it)
        writer.add_scalar('val/loss_v', avg_loss_v, it)
        writer.add_scalar('val/loss_bond_aux', avg_loss_bond_aux, it)
        writer.add_scalar('val/loss_bond_final', avg_loss_bond_final, it)
        writer.add_scalar('val/loss_bond_dist', avg_loss_bond_dist, it)
        writer.add_scalar('val/loss_bond_angle', avg_loss_bond_angle, it)
        writer.add_scalar('val/loss_torsion_angle', avg_loss_torsion_angle, it)
        writer.flush()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        for it in range(start_iter, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
