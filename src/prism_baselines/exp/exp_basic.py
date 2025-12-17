# exp_basic.py
# Licensed under the MIT License

# Parts of this code have been adapted from the following sources: 

# https://github.com/thuml/Neural-Solver-Library, Neural-Solver-Library/exp/exp_basic.py
# Modified: adapted to data structure and outputs, SDF calculated directly from geometry
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


import os
import torch
from neural_solver_library.models.model_factory import get_model
from prism_baselines.data_provider.data_factory import get_data

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}", flush=True)
    return total_params


class Exp_Basic(object):
    def __init__(self, args):
        print("Starting script...", flush=True)
        print(f"args.output_types = {args.output_types}", flush=True)
        self.dataset, self.train_loader, self.val_loader, self.test_loader, self.real_loader, args.shapelist = get_data(args)
        print("After dataloading...", flush=True)
        self.model = get_model(args).cuda()
        self.args = args
        print(self.args, flush=True)
        print(self.model, flush=True)
        count_parameters(self.model)

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
