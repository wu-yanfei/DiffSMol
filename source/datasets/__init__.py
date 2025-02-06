import torch
from torch.utils.data import Subset
from .shape_mol_dataset import ShapeMolDataset
from .shape_data import ShapeDataset
import pdb
import numpy as np

def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'shapemol':
        dataset = ShapeMolDataset(config, *args, **kwargs)
    elif name == 'shape':
        dataset = ShapeDataset(config, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config and config.dataset != 'moses2':
        split = torch.load(config.split)
        subsets = {}
        dataset._connect_db()
        
        for k, v in split.items():
            v = [idx for idx in v if idx < dataset.size]
            if k == 'train':
                random_valid_indices = np.random.choice(v, 1000).tolist()
                v = [idx for idx in range(dataset.size) if idx not in random_valid_indices]
                subsets['valid'] = Subset(dataset, indices=random_valid_indices)

            subsets[k] = Subset(dataset, indices=v)
        return dataset, subsets
    elif 'split' in config and config.dataset == 'moses2':
        subsets = {}
        dataset._connect_db()
        split = torch.load(config.split)
        for k, v in split.items():
            v = [idx for idx in v if idx < dataset.size]
            if k == 'train':
                random_valid_indices = np.random.choice(v, 1000).tolist()
                v = [idx for idx in range(dataset.size) if idx not in random_valid_indices]
                subsets['valid'] = Subset(dataset, indices=random_valid_indices)
            else:
                continue
            subsets[k] = Subset(dataset, indices=v)
        #split = torch.load(config.split)
        #subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
