# run.py
# Runs surrogate models
# Licensed under the MIT License

# Parts of this code have been adapted from the following sources: 

# https://github.com/thuml/Neural-Solver-Library, Neural-Solver-Library/run.py
# Modified: additional input arguments
# Licensed under the MIT License
# © Original Author(s): Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long

import os, sys
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for parent in (p, *p.parents):
        if (parent / ".git").exists() or (parent / "UPSTREAM.md").exists() or (parent / "requirements.txt").exists():
            return parent
    return start.resolve().parents[2]

HERE = Path(__file__).parent
ROOT = _find_repo_root(HERE)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party"))

# -----------------------------------------------

import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

parser = argparse.ArgumentParser('Training Neural PDE Solvers')

parser.add_argument('--base_dir', type=str, default='./experiments', help='Directory to store all outputs')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint if available')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='maximum epochs')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')
parser.add_argument('--pct_start', type=float, default=0.3, help='oncycle lr schedule')
parser.add_argument('--batch-size', type=int, default=8, help='batch size')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
parser.add_argument('--prefetch_factor', type=int, default=1, help='prefetch factor')
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument('--max_grad_norm', type=float, default=None, help='make the training stable')
parser.add_argument('--derivloss', type=bool, default=False, help='adopt the spatial derivate as regularization')
parser.add_argument('--teacher_forcing', type=int, default=1,
                    help='adopt teacher forcing in autoregressive to speed up convergence')
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer type, select from Adam, AdamW')
parser.add_argument('--scheduler', type=str, default='OneCycleLR',
                    help='learning rate scheduler, select from [OneCycleLR, CosineAnnealingLR, StepLR]')
parser.add_argument('--step_size', type=int, default=100, help='step size for StepLR scheduler')
parser.add_argument('--gamma', type=float, default=0.5, help='decay parameter for StepLR scheduler')
parser.add_argument('--output_types', type=str, default='all', help='type of output')

## data
parser.add_argument('--data_path', type=str, default='/data/fno/', help='data folder')
parser.add_argument('--c_norm', type=str, default='coef_norm.npy', help='coefficient normalization file')
parser.add_argument('--max_neighbors', type=int, default=32, help='max number of neighbors in graph')
parser.add_argument('--max_pc_points', type=int, default=8192, help='maximum number of points to represent shape')
parser.add_argument('--max_points', type=int, default=0, help='maximum number of points for training')
parser.add_argument('--return_shape', type=int, default=0, help='return shape point cloud or not')
parser.add_argument('--use_orig_mesh', type=int, default=0, help='use entire mesh points or subset')
parser.add_argument('--loader', type=str, default='airfoil', help='type of data loader')
parser.add_argument('--train_ratio', type=float, default=0.8, help='training data ratio')
parser.add_argument('--ntrain', type=int, default=1000, help='training data numbers')
parser.add_argument('--ntest', type=int, default=200, help='test data numbers')
parser.add_argument('--normalize', type=bool, default=False, help='make normalization to output')
parser.add_argument('--norm_type', type=str, default='UnitTransformer',
                    help='dataset normalize type. select from [UnitTransformer, UnitGaussianNormalizer]')
parser.add_argument('--geotype', type=str, default='unstructured',
                    help='select from [unstructured, structured_1D, structured_2D, structured_3D]')
parser.add_argument('--time_input', type=bool, default=False, help='for conditional dynamic task')
parser.add_argument('--space_dim', type=int, default=2, help='position information dimension')
parser.add_argument('--fun_dim', type=int, default=0, help='input observation dimension')
parser.add_argument('--out_dim', type=int, default=1, help='output observation dimension')
parser.add_argument('--shapelist', type=list, default=None, help='for structured geometry')
parser.add_argument('--downsamplex', type=int, default=1, help='downsample rate in x-axis')
parser.add_argument('--downsampley', type=int, default=1, help='downsample rate in y-axis')
parser.add_argument('--downsamplez', type=int, default=1, help='downsample rate in z-axis')
parser.add_argument('--radius', type=float, default=0.2, help='for construct geometry')


## task
parser.add_argument('--task', type=str, default='steady',
                    help='select from [steady, dynamic_autoregressive, dynamic_conditional]')
parser.add_argument('--T_in', type=int, default=10, help='for input sequence')
parser.add_argument('--T_out', type=int, default=10, help='for output sequence')


## models
parser.add_argument('--model', type=str, default='Transolver')
parser.add_argument('--n_hidden', type=int, default=64, help='hidden dim')
parser.add_argument('--n_layers', type=int, default=3, help='layers')
parser.add_argument('--n_heads', type=int, default=4, help='number of heads')
parser.add_argument('--act', type=str, default='gelu')
parser.add_argument('--mlp_ratio', type=int, default=1, help='mlp ratio for feedforward layers')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--unified_pos', type=int, default=0, help='for unified position embedding')
parser.add_argument('--ref', type=int, default=8, help='number of reference points for unified pos embedding')

## model specific configuration
parser.add_argument('--slice_num', type=int, default=32, help='number of physical states for Transolver')
parser.add_argument('--modes', type=int, default=12, help='number of basis functions for LSM and FNO')
parser.add_argument('--psi_dim', type=int, default=8, help='number of psi_dim for ONO')
parser.add_argument('--attn_type', type=str, default='nystrom',help='attn_type for ONO, select from nystrom, linear, selfAttention')
parser.add_argument('--mwt_k', type=int, default=3,help='number of wavelet basis functions for MWT')

## eval
parser.add_argument('--eval', type=int, default=0, help='evaluation or not')
parser.add_argument('--test_type', type=int, default=0, help='evaluation type - include all test sets or just basic set')
parser.add_argument('--save_name', type=str, default='Transolver_check', help='name of folders')
parser.add_argument('--vis_num', type=int, default=10, help='number of visualization cases')
parser.add_argument('--vis_bound', type=int, nargs='+', default=None, help='size of region for visualization, in list')

args = parser.parse_args()
eval = args.eval
save_name = args.save_name
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    elif args.task == 'steady_design_buildings':
        from prism_baselines.exp.exp_steady_design_buildings import Exp_Steady_Design_Buildings
        exp = Exp_Steady_Design_Buildings(args)
    else:
        raise NotImplementedError

    if eval:
        if(args.test_type==0):
            exp.test()
        else:
            exp.test_all()
    else:
        exp.train()
        exp.test()


if __name__ == "__main__":
    main()
