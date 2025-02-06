import os
import oddt
import pickle
import lmdb
import time
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import copy
from utils.data import *
from .shape_mol_data import ShapeMolData, torchify_dict
from utils.shape import *
from functools import partial
from multiprocessing import Pool
from utils.subproc_shapeAE import SubprocShapeAE

class ShapeMolDataset(Dataset):

    def __init__(self, config, transform):
        super().__init__()
        self.config = config
        self.raw_path = config.path.rstrip('/')
        self.processed_dir = config.processed_path
        if config.dataset == 'crossdocked': self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(self.processed_dir,
                                           config.dataset + f'_processed_{config.version}.lmdb')
        self.transform = transform
        self.db = None
        self.keys = None
        self.skip_idxs = []
        self.shape_type = config.shape.shape_type
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=self.config.datasize*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
            self.size = len(self.keys)
        

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=self.config.datasize*(1024*1024*1024),   # 20GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        if self.config.dataset == 'crossdocked':
            self._process_crossdock(db)
        elif self.config.dataset == 'moses2':
            self._process_mose(db)
        elif self.config.dataset == 'geom-drugs':
            self._process_geom(db)

    def _process_mose(self, db):
        shape_func, subproc_voxelae = get_shape_func(self.config.shape)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            if 'test' in self.raw_path:
                all_mols = pickle.load(open(self.raw_path, 'rb'))
            else:
                all_mols = pickle.load(open(self.raw_path, 'rb'))['rdkit_mol_cistrans_stereo']
            
            batch = self.config.shape.batch_size * self.config.shape.num_workers
            chunk_size = self.config.chunk_size
            
            for chunk_id, i in enumerate(range(0, len(all_mols), chunk_size)):
                print(f'processing chunk {chunk_id}....')
                chunk_mols = all_mols[i:min(len(all_mols), i+chunk_size)]

                pool = Pool(processes=self.config.num_workers)
                chunk_dicts = []
                for data in tqdm(pool.imap(parse_rdkit_mol, chunk_mols)):
                    chunk_dicts.append(data)
                pool.close()
                print("finish rdkit parse")
                
                for j in tqdm(range(i, min(len(all_mols), i+chunk_size), batch)):
                    batch_mols = all_mols[j:min(j+batch, len(all_mols))]
                    batch_dicts = chunk_dicts[j-i:min(j+batch, len(all_mols))-i]
                    
                    if len(batch_mols) == 0: continue
                    if self.config.shape.use_shape:
                        remove_idxs, batch_shape_embs, batch_bounds, batch_pointclouds, batch_pointcloud_centers = shape_func(batch_mols)
                    
                    for k, ligand_dict in enumerate(batch_dicts):
                        #try:
                        data = ShapeMolData.from_ligand_dicts(
                            ligand_dict=torchify_dict(ligand_dict),
                        )
                        if self.config.shape.use_shape:
                            data.shape_emb = batch_shape_embs[k]
                            data.ligand_pos = data.ligand_pos - batch_pointcloud_centers[k]
                        
                        data.bound = batch_bounds[k]

                        if 'test' in self.raw_path:
                            data.point_cloud = batch_pointclouds[k]
                            data.mol = batch_mols[k]

                        data.ligand_index = torch.tensor(i+j+k)
                        data = data.to_dict()  # avoid torch_geometric version issue
                        
                        txn.put(
                            key=str(i+j+k).encode(),
                            value=pickle.dumps(data)
                        )
            
            if self.config.shape.shape_parallel: subproc_voxelae.close()
        db.close()

    def _process_geom(self, db):
        shape_func, subproc_voxelae = get_shape_func(self.config.shape)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            mols_path = self.raw_path + "/" + "GEOM_mols.pkl"
            print(mols_path)
            if os.path.exists(mols_path):
                all_mols = pickle.load(open(mols_path, 'rb'))
            else:
                files = [x for x in os.listdir(self.raw_path) if 'pickle' in x]
                all_mols = []
                for f in tqdm(files):
                    f_file = open(self.raw_path + "/" + f, 'rb')
                    tmp = pickle.load(f_file)['conformers'][0]['rd_mol']
                    f_file.close()
                    all_mols.append(tmp)
                with open(mols_path, 'wb') as f:
                    pickle.dump(all_mols, f)
            
            batch = self.config.shape.batch_size * self.config.shape.num_workers
            chunk_size = self.config.chunk_size
            overall_idx = 0
            for chunk_id, i in enumerate(range(0, len(all_mols), chunk_size)):
                print(f'processing chunk {chunk_id}....')
                chunk_mols = all_mols[i:min(len(all_mols), i+chunk_size)]

                pool = Pool(processes=self.config.num_workers)
                chunk_dicts = []
                for data in tqdm(pool.imap(parse_rdkit_mol, chunk_mols)):
                    chunk_dicts.append(data)
                pool.close()
                print("finish rdkit parse")
                
                for j in tqdm(range(i, min(len(all_mols), i+chunk_size), batch)):
                    batch_mols = all_mols[j:min(j+batch, len(all_mols))]
                    batch_dicts = chunk_dicts[j-i:min(j+batch, len(all_mols))-i]
                    
                    if len(batch_mols) == 0: continue
                    if self.config.shape.use_shape:
                        remove_idxs, batch_shape_embs, batch_bounds, batch_pointclouds, batch_pointcloud_centers = shape_func(batch_mols)
                    else:
                        remove_idxs = []

                    offset = 0
                    for k, ligand_dict in enumerate(batch_dicts):
                        #try:
                        if k in remove_idxs:
                            offset += 1
                            continue
                        data = ShapeMolData.from_ligand_dicts(
                            ligand_dict=torchify_dict(ligand_dict),
                        )
                        if self.config.shape.use_shape:
                            data.shape_emb = batch_shape_embs[k-offset]
                        data.ligand_pos = data.ligand_pos - batch_pointcloud_centers[k-offset]
                        
                        data.bound = batch_bounds[k-offset]

                        if 'test' in self.raw_path:
                            data.point_cloud = batch_pointclouds[k-offset]
                            data.mol = batch_mols[k-offset]

                        data.ligand_index = torch.tensor(i+j+k)
                        data = data.to_dict()  # avoid torch_geometric version issue
                        
                        txn.put(
                            key=str(overall_idx).encode(),
                            value=pickle.dumps(data)
                        )
                        overall_idx += 1
            
            if self.config.shape.shape_parallel: subproc_voxelae.close()
        db.close()

    def _process_crossdock(self, db):
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)
        shape_func, subproc_voxelae = get_shape_func(self.config.shape)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            ligand_dicts = []
            all_mols = []
            skip_num = 0
            #for i, (protein_fn, ligand_fn, _) in enumerate(tqdm(index)):
            #    tmp = "/"+"/".join(ligand_fn.split("/")[2:])
            #    #try:
            #    protein_fn, ligand_fn, ligand_dict, pocket_dict, rdmol = parse_sdf_mol_with_pdb(index[i], raw_path = self.raw_path)
            #    #    except Exception as e:
            #    #        print(e)
            #    #        print("skip %d molecules at %d" % (skip_num, i))
            #    #        skip_num += 1
            #    #        continue
            #    ligand_dicts.append((ligand_dict, protein_fn, ligand_fn))
            #    all_mols.append(rdmol)
            #index = index[:10000]
            pool = Pool(processes=self.config.num_workers)
            all_mols, ligand_dicts = [], []
            parse_func = partial(parse_sdf_mol_with_pdb, raw_path=self.raw_path)
            skip_num = 0
            for protein_fn, ligand_fn, ligand_dict, pocket_dict, rdmol in tqdm(pool.imap(parse_func, index)):
                if protein_fn is None or rdmol is None:
                    skip_num += 1
                else:
                    ligand_dicts.append((ligand_dict, pocket_dict, protein_fn, ligand_fn))
                    all_mols.append(rdmol)

            pool.close()
            print("skip %d molecules in total" % (skip_num))
            print("get %d processed molecules" % (len(all_mols)))
            
            batch = self.config.shape.batch_size * self.config.shape.num_workers
            for batch_id, i in enumerate(tqdm(range(0, len(all_mols), batch))):
                batch_mols = all_mols[i:min(len(all_mols), i+batch)]
                batch_ligand_dicts = ligand_dicts[i:min(len(all_mols), i+batch)]

                                
                if self.config.shape.use_shape:
                    remove_idxs, batch_shape_embs, batch_bounds, batch_pointclouds, batch_pointcloud_centers = shape_func(batch_mols)
                
                if batch_shape_embs is None: continue
                offset = 0
                for j, (ligand_dict, pocket_dict, protein_fn, ligand_fn) in enumerate(batch_ligand_dicts):
                    try:
                        if j in remove_idxs:
                            offset += 1
                            raise ValueError("skip %s due to mesh" % (ligand_fn))
                        
                        data = ShapeMolData.from_ligand_dicts(
                            ligand_dict=torchify_dict(ligand_dict),
                            protein_dict=torchify_dict(pocket_dict)
                        )
                        
                        if self.config.shape.use_shape:
                            data.shape_emb = batch_shape_embs[j-offset]
                        data.ligand_pos = data.ligand_pos - batch_pointcloud_centers[j-offset]
                        data.protein_pos = data.protein_pos - batch_pointcloud_centers[j-offset] - data.ligand_center
                        
                        data.bound = batch_bounds[j-offset]

                        if 'test' in self.raw_path:
                            data.point_cloud = batch_pointclouds[j-offset]
                            data.point_cloud_center = batch_pointcloud_centers[j-offset]
                            data.mol = batch_mols[j]
                        data.protein_filename = protein_fn
                        data.ligand_filename = ligand_fn
                        data = data.to_dict()  # avoid torch_geometric version issue
                        txn.put(
                            key=str(i+j).encode(),
                            value=pickle.dumps(data)
                        )
                    except Exception as e:
                        print(e)
                        num_skipped += 1
                        print('Skipping (%d) %s' % (num_skipped, ligand_fn, ))
                        continue
        db.close()
    
    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        if idx in self.skip_idxs:
            new_idx = np.random.choice(self.size, 1)
            while new_idx in self.skip_idxs:
                new_idx = np.random.choice(self.size, 1)
            return self.__getitem__(new_idx)

        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ShapeMolData(**data)
        data.id = idx
        
        shape_emb = data.shape_emb
        if self.transform is not None:
            try:
                data = self.transform(data)
            except:
                self.skip_idxs.append(idx)
                new_idx = np.random.choice(self.size, 1)[0]
                while new_idx in self.skip_idxs:
                    new_idx = np.random.choice(self.size, 1)[0]
                return self.__getitem__(new_idx)
        
        data.shape_emb = shape_emb
        return data
        

def get_shape_func(config):
    if config.shape_type == 'electroshape':
        shape_func = get_electro_shape_emb
    elif config.shape_type == 'voxelAE_shape':
        atom_stamp = get_atom_stamp(grid_resolution=config.grid_resolution, max_dist=4)
        if config.shape_parallel: shapeae = SubprocShapeAE(config)
        else: shapeae = build_voxel_shapeAE_model(config, device='cuda')
        
        shape_func = partial(get_voxelAE_shape_emb,
                             model=shapeae,
                             atom_stamp=atom_stamp,
                             grid_resolution=config.grid_resolution,
                             max_dist=config.max_dist,
                             batch_size=config.batch_size,
                             shape_parallel=config.shape_parallel
                             )
    elif config.shape_type == 'pointAE_shape':
        if config.shape_parallel: shapeae = SubprocShapeAE(config)
        else: shapeae = build_point_shapeAE_model(config, device='cuda')
        shape_func = partial(get_pointAE_shape_emb,
                             model=shapeae,
                             point_cloud_samples=config.point_cloud_samples,
                             config=config,
                             batch_size=config.batch_size,
                             shape_parallel=config.shape_parallel
                             )
    return shape_func, shapeae
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = ShapeMolDataset(args.path)
    print(len(dataset), dataset[0])
