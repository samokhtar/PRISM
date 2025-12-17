import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data.dataloader import default_collate
from torch_geometric import nn as nng
from torch_geometric.data import Data


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

class building_design(object):
    def __init__(self, args):
        self.data_path = args.data_path
        c_norm = args.c_norm
        self.radius = args.radius
        self.max_neighbors = args.max_neighbors
        self.max_shape_points = args.max_pc_points
        self.max_points = args.max_points
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.prefetch_factor = args.prefetch_factor
        self.return_shape = bool(args.return_shape)
        self.use_orig_mesh = bool(args.use_orig_mesh)
        self.m_ext = '_1M' if(not self.use_orig_mesh) else ''
        self.subpath = '/Performance/CFD_VTK_processed/01_Buildings/0000_0499'
        self.subpath_real = '/Performance/CFD_VTK_processed/01_Buildings/realbuildings'
        
        self.vtknpy_dir = args.data_path + self.subpath + self.subpath
        self.vtknpy_real_dir = args.data_path + self.subpath_real
        self.real_root = Path(self.vtknpy_real_dir)
        self.real_ids = np.unique([(str(p.relative_to(self.real_root)), p.relative_to(self.real_root).name) for p in self.real_root.rglob('*') if p.is_dir() and len(p.relative_to(self.real_root).parts) == 1])
        self.real_ids = [r+'_0' for r in self.real_ids]
        
        self.train_test_param_file = 'traintestParam_5000.json'
        print(f"args.output_types = {args.output_types}", flush=True)

        self.coef_norm = [[0,0,45,1250,0,0,0],[1250,1250,175,1100,0.6,0.6,0.4],[0,0,0,0,0],[4,4,4,50,0.5]]

        dataset = pd.read_csv(self.vtknpy_dir + '/' + 'VTK_datasetStats'+self.m_ext+'.csv')
        self.dataset = dataset[dataset['all']==1]
        self.d_samples = self.dataset['b_type'].values
        
        split_len, split_ids = {}, {}
        self.train_test_splits = json.loads(open(os.path.join(self.data_path, 'Parameters') + '/' + self.train_test_param_file).read())
        for k in self.train_test_splits.keys():
            subKey = np.array(self.train_test_splits[k])
            inc_idx = self.dataset[self.dataset['idx'].isin(subKey)]
            split_len[k] = len(inc_idx)
            split_ids[k] = np.array(inc_idx['b_name'])
        self.split_len = split_len
        self.split_ids = split_ids
        print('splits', split_len, flush=True)

    def get_samples(self):
        return self.d_samples

    def get_folds(self, folds=['train','val','test','real']):
        f_datasets = []
        for f in folds:
            cur_root = self.vtknpy_dir if(f != 'real') else self.vtknpy_real_dir
            cur_samples = self.split_ids[f] if(f != 'real') else self.real_ids
            f_dataset = BuildingCFDDataset(
                cur_root,
                cur_samples,
                coef_norm=self.coef_norm,
                r=self.radius,
                max_neighbors=self.max_neighbors, 
                max_points=self.max_points, 
                max_pc_points=self.max_shape_points, 
                return_shape=self.return_shape,
                use_orig_mesh = self.use_orig_mesh
            )
            f_datasets.append(f_dataset)
        return f_datasets

    def get_loader(self, folds=['train','val','test','real']):
        datafold = self.get_folds(folds)
        persistent_workers = True if(self.num_workers>1) else False
        d_f_ar = []
        for i in range(len(folds)):
            shuffle = True if(folds[i]=='train') else False
            dl_ml = torch.utils.data.DataLoader(datafold[i], 
                                                batch_size=self.batch_size, 
                                                drop_last=False, 
                                                pin_memory=True, 
                                                num_workers=self.num_workers, 
                                                persistent_workers=persistent_workers, 
                                                collate_fn=safe_collate,
                                                shuffle=shuffle, 
                                                prefetch_factor=self.prefetch_factor)
            d_f_ar.append(dl_ml)
        l_sets = [self.split_len[k] for k in self.train_test_splits.keys()]
        l_sets.append(self.real_ids)
        d_f_ar.append(l_sets)
        return d_f_ar

class BuildingCFDDataset(torch.utils.data.Dataset):
    def __init__(self, root, samples, coef_norm=None, r = 2, max_neighbors=32, max_points = 0, max_pc_points=8192, return_shape=False, surface_pts_only=False, use_orig_mesh=False):
        self.root = root
        self.samples = samples
        self.coef_norm = coef_norm
        self.r = r
        self.max_neighbors = max_neighbors
        self.max_points = max_points
        self.max_shape_points = max_pc_points
        self.return_shape = return_shape
        self.surface_pts_only = surface_pts_only
        self.use_orig_mesh = use_orig_mesh
        self.m_ext = '_1M' if(not use_orig_mesh) else ''
        
        print('use_orig_mesh',self.use_orig_mesh, flush=True)
        print('m_ext',self.m_ext, flush=True)
        print('self.coef_norm',self.coef_norm, flush=True)
        
        if self.coef_norm is not None:
            mean_in, std_in, mean_out, std_out = self.coef_norm
            self.mean_in = torch.tensor(mean_in, dtype=torch.float32)
            self.std_in = torch.tensor(std_in, dtype=torch.float32)
            self.mean_out = torch.tensor(mean_out, dtype=torch.float32)
            self.std_out = torch.tensor(std_out, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        x_dir = self.root + '/' + '_'.join(s.split('_')[0:2])
        y_dir = self.root + '/' + '_'.join(s.split('_')[0:2]) + '/' + s
        if (not (os.path.exists(x_dir) and os.path.exists(y_dir) )):
            print(f"Preprocessed folder(s) missing: {x_dir}, {y_dir}", flush=True)
            return None
        else:
            missing_files = [f for f in [os.path.join(x_dir, 'surf' + self.m_ext + '.npy'),os.path.join(y_dir, 'pos_rot' + self.m_ext + '.npy'),os.path.join(y_dir, 'x_rot' + self.m_ext + '.npy'),os.path.join(y_dir, 'y' + self.m_ext + '.npy')] if not os.path.exists(f)]
            if missing_files:
                print(f"Missing files for sample {s}:", flush=True)
                for f in missing_files:
                    print(f" - {f}", flush=True)
                return None  
            
            surf = torch.from_numpy(np.load(os.path.join(x_dir, 'surf'+self.m_ext+'.npy')).astype(np.float32).copy())
            if(self.max_points == 0):
                pos = torch.from_numpy(np.load(os.path.join(y_dir, 'pos_rot'+self.m_ext+'.npy')).astype(np.float32).copy())
                x = torch.from_numpy(np.load(os.path.join(y_dir, 'x_rot'+self.m_ext+'.npy')).astype(np.float32).copy())
                y = torch.from_numpy(np.load(os.path.join(y_dir, 'y'+self.m_ext+'.npy')).astype(np.float32).copy())
                n_surf, n_pos, n_x, n_y = len(surf), len(pos), len(x), len(y)
                if not (n_surf == n_pos == n_x == n_y):
                    print(f"[{s}] Array length mismatch: surf={n_surf}, pos={n_pos}, x={n_x}, y={n_y}", flush=True)
                    return None 
            else:
                rand_idx = np.random.randint(0, surf.shape[0], size=self.max_points)
                pos = np.load(os.path.join(y_dir, 'pos_rot'+self.m_ext+'.npy'))
                x = np.load(os.path.join(y_dir, 'x_rot'+self.m_ext+'.npy'))
                y = np.load(os.path.join(y_dir, 'y'+self.m_ext+'.npy'))
                n_surf, n_pos, n_x, n_y = len(surf), len(pos), len(x), len(y)
                if not (n_surf == n_pos == n_x == n_y):
                    print(f"[{s}] Array length mismatch: surf={n_surf}, pos={n_pos}, x={n_x}, y={n_y}", flush=True)
                    return None 
                surf = surf[rand_idx]
                pos = torch.from_numpy(pos[rand_idx].astype(np.float32).copy())
                x = torch.from_numpy(x[rand_idx].astype(np.float32).copy())
                y = torch.from_numpy(y[rand_idx].astype(np.float32).copy())
                
            for name, arr in [('x', x), ('y', y), ('pos', pos), ('surf', surf)]:
                if torch.isnan(arr).any() or torch.isinf(arr).any():
                    print(f"Invalid values detected in {name} for sample {s}", flush=True)
                    return None  

            if self.coef_norm is not None:  
                x = (x - self.mean_in.to(x.device)) / (self.std_in.to(x.device) + 1e-8)
                y = (y - self.mean_out.to(x.device)) / (self.std_out.to(x.device) + 1e-8)

            data = Data(pos=pos, x=x, y=y, surf=surf.bool())
            data = create_edge_index_radius(data, self.r, self.max_neighbors)
            
            if(self.return_shape):
                shape_pc = get_shape(data, max_n_point=self.max_shape_points)
                print(shape_pc.shape, flush=True)
                return data.pos, data.x, data.y, data.surf, data.edge_index, shape_pc
            else:
                return data.pos, data.x, data.y, data.surf, data.edge_index
            
def get_shape(data, max_n_point=8192):
    surf_indices = torch.where(data.surf)[0].tolist()
    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))
    shape_pc = data.pos[surf_indices].clone()
    return shape_pc

def create_edge_index_radius(data, r, max_neighbors=32):
    t0 = time.time()
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data