# exp_steady_design_buildings.py
# Licensed under the MIT License

# Parts of this code have been adapted from the following sources: 

# https://github.com/thuml/Neural-Solver-Library, Neural-Solver-Library/exp/exp_steady_design.py
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

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_baselines.exp.exp_basic import Exp_Basic
from neural_solver_library.utils.logger import Logger


def compute_all_errors(pred, gt):
    pred = pred.squeeze(0)
    gt = gt.squeeze(0)
    errors = {}

    diff = pred - gt
    abs_diff = torch.abs(diff)
    squared_diff = diff ** 2

    errors["mse"] = squared_diff.mean(dim=0)
    errors["mae"] = abs_diff.mean(dim=0)
    errors["rmse"] = torch.sqrt(errors["mse"])
    errors["maxae"] = torch.max(abs_diff, dim=0).values
    errors["medae"] = torch.median(abs_diff, dim=0).values
    errors["p95ae"] = torch.quantile(abs_diff, 0.95, dim=0)
    errors["mape"] = (abs_diff / (gt.abs().clamp(min=0.05))).mean(dim=0)
    errors["medape"] = torch.median(abs_diff / (gt.abs().clamp(min=0.05)).clamp(min=0.05), dim=0).values
    errors["p95ape"] = torch.quantile(abs_diff / (gt.abs().clamp(min=0.05)).clamp(min=0.05), 0.95, dim=0)
    errors["rel_l2"] = torch.norm(diff) / (torch.norm(gt.clamp(min=0.05)).clamp(min=0.05))
    errors["rel_rmse"] = errors["rmse"] / (gt.abs().clamp(min=0.05).mean(dim=0)).clamp(min=0.05)

    if pred.shape[-1] == 3:
        cos_sim = F.cosine_similarity(pred, gt, dim=1)
        errors["cosine_similarity"] = cos_sim.mean()
        angle = torch.acos(torch.clamp(cos_sim, -1.0, 1.0))
        errors["angular_error_rad"] = angle.mean()

    ss_res = squared_diff.sum(dim=0)
    ss_tot = ((gt - gt.mean(dim=0)) ** 2).sum(dim=0)
    errors["r2"] = 1 - (ss_res / (ss_tot + 1e-8))

    if pred.shape[-1] == 1 or len(pred.shape) == 1:
        sign_diff = (torch.sign(pred) != torch.sign(gt)).float()
        errors["sign_error_rate"] = sign_diff.mean()

    return errors

def aggregate_metrics(metric_list, apply_sqrt=False):
    numeric_keys = [k for k in metric_list[0].keys() if isinstance(metric_list[0][k], (int, float, np.ndarray))]
    aggregated = {}
    for key in numeric_keys:
        values = [m[key] for m in metric_list if key in m]
        if isinstance(values[0], np.ndarray):
            stacked = np.stack(values)
            mean_val = np.mean(stacked, axis=0)
            if apply_sqrt:
                mean_val = np.sqrt(mean_val)
            aggregated[key] = mean_val
        else:
            mean_val = np.mean(values)
            if apply_sqrt:
                mean_val = np.sqrt(mean_val)
            aggregated[key] = mean_val
    return aggregated



class Exp_Steady_Design_Buildings(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady_Design_Buildings, self).__init__(args)

    def vali(self, log_file=None):
        myloss = nn.MSELoss(reduction='none')
        self.model.eval()
        rel_err = 0.0
        index = 0
        mean = torch.tensor(self.val_loader.dataset.coef_norm[2], dtype=torch.float32).cuda()
        std = torch.tensor(self.val_loader.dataset.coef_norm[3], dtype=torch.float32).cuda()

        with torch.no_grad():
            
            for batch in self.val_loader:
                torch.cuda.empty_cache()
                if batch is None:
                    continue
                pos, fx, y, surf, geo = batch
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if(geo.shape[0]==1):
                    geo = geo.squeeze(0)
                if self.args.fun_dim == 0:
                    fx = None
                out = self.model(x, fx, geo=geo)
                
                pred_velo = out[..., :-1][~surf]
                gt_velo = y[..., :-2][~surf] 
                
                mask = torch.norm(y[..., :-2][~surf] * std[:-2] + mean[:-2], dim=-1) > 0.1
                if mask.sum() == 0:
                    print(f"Skipping index {index} due to empty velocity mask", flush=True)
                    continue
                    
                pred_velo = pred_velo[mask]
                gt_velo = gt_velo[mask] 

                loss_spd = myloss(torch.norm(pred_velo, dim=-1), torch.norm(gt_velo, dim=-1)).mean()
                loss_pressC = myloss(out[..., -1][surf], y[..., -1][surf]).mean()
                loss_velo_var = myloss(torch.clamp(torch.norm(pred_velo, dim=-1), min=0.0), torch.clamp(torch.norm(gt_velo, dim=-1), min=0.0)).mean()
                del pred_velo, gt_velo
                
                if(self.args.output_types == 'spd'):
                    loss = loss_spd
                elif(self.args.output_types == 'vel'):
                    loss = loss_velo_var
                elif(self.args.output_types == 'pres'):
                    loss = loss_pressC
                else:
                    loss = (0.5 * loss_velo_var) + (0.5 * loss_spd) + (0.05 * loss_pressC)

                print('val','press',loss_pressC.detach().cpu().numpy(),'vel',loss_velo_var.detach().cpu().numpy(),'spd',loss_spd.detach().cpu().numpy(),'tot',loss.detach().cpu().numpy(), flush=True)
                
                rel_err += loss.item()

                index += 1

        rel_err /= float(index)
        if log_file:
            log_file.write(f"[Validation] Relative Error: {rel_err:.6f}\n")
        return rel_err

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        base_dir = './experiments'
        logger = Logger(base_dir, self.args.save_name, self.args)

        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise ValueError('Unsupported optimizer')

        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.args.lr,
                epochs=self.args.epochs, steps_per_epoch=len(self.train_loader),
                pct_start=self.args.pct_start
            )
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        else:
            raise ValueError('Unsupported scheduler')

        myloss = nn.MSELoss(reduction='none')
        start_epoch = 0
        
        mean = torch.tensor(self.train_loader.dataset.coef_norm[2], dtype=torch.float32).cuda()
        std = torch.tensor(self.train_loader.dataset.coef_norm[3], dtype=torch.float32).cuda()

        # Check if we want to resume from checkpoint
        ckpt_path = os.path.join(base_dir, self.args.save_name, 'checkpoint_latest.pt')
        print('ckpt_path',ckpt_path, flush=True)
        print('ckpt_path_exists',os.path.exists(ckpt_path), flush=True)
        print('attribute',getattr(self.args, 'resume', False), flush=True)
        if os.path.exists(ckpt_path) and getattr(self.args, 'resume', False):
            print(f"Resuming training from checkpoint: {ckpt_path}", flush=True)
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Checkpoint resumed from epoch {start_epoch}", flush=True)

        for ep in range(start_epoch, self.args.epochs):
            self.model.train()
            train_loss = 0.0
            index = 0

            for batch in self.train_loader:
                if batch is None:
                    continue
                pos, fx, y, surf, geo = batch
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(geo).any():
                    print(f"NaNs in inputs at epoch {ep}, batch {index}",flush=True)
                    continue
                if fx is not None and torch.isnan(fx).any():
                    print(f"NaNs in fx at epoch {ep}, batch {index}",flush=True)
                    continue
                
                if(geo.shape[0]==1):
                    geo = geo.squeeze(0)

                if self.args.fun_dim == 0:
                    fx = None
                    
                gt_velo = y[..., :-2][~surf]          
                mask = torch.norm(gt_velo * std[:-2] + mean[:-2], dim=-1) > 0.1
                if mask.sum() == 0:
                    print(f"Skipping index {index} due to empty velocity mask", flush=True)
                    continue
                    
                out = self.model(x, fx, geo=geo)
                if torch.isnan(out).any():
                    print(f"NaNs in model output at epoch {ep}, batch {index}",flush=True)
                    continue
                
                pred_velo = out[..., :-1][~surf]
                    
                pred_velo = pred_velo[mask]
                gt_velo = gt_velo[mask] 

                loss_spd = myloss(torch.norm(pred_velo, dim=-1), torch.norm(gt_velo, dim=-1)).mean()
                loss_pressC = myloss(out[..., -1][surf], y[..., -1][surf]).mean()
                loss_velo_var = myloss(torch.clamp(torch.norm(pred_velo, dim=-1), min=0.0), torch.clamp(torch.norm(gt_velo, dim=-1), min=0.0)).mean()
                del pred_velo, gt_velo
                
                if(self.args.output_types == 'spd'):
                    loss = loss_spd
                elif(self.args.output_types == 'vel'):
                    loss = loss_velo_var
                elif(self.args.output_types == 'pres'):
                    loss = loss_pressC
                else:
                    if torch.isnan(loss_spd) or torch.isnan(loss_velo_var) or torch.isnan(loss_pressC):
                        del loss_spd, loss_velo_var, loss_pressC
                        print(f"NaN in loss terms at epoch {ep}, batch {index}",flush=True)
                        del out 
                        torch.cuda.empty_cache()
                        continue
                    loss = (0.5 * loss_velo_var) + (0.5 * loss_spd) + (0.05 * loss_pressC)
                    
                print('train',ep,'press',loss_pressC.detach().cpu().numpy()*0.05,'vel',loss_velo_var.detach().cpu().numpy()*0.5,'spd',loss_spd.detach().cpu().numpy()*0.5,'tot',loss.detach().cpu().numpy(), flush=True)
                
                optimizer.zero_grad()
                loss.backward()
                if self.args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                if self.args.scheduler == 'OneCycleLR':
                    scheduler.step()

                train_loss += loss.item()
                index += 1

            if self.args.scheduler in ['StepLR', 'CosineAnnealingLR']:
                scheduler.step()

            train_loss /= float(index)
            val_err = self.vali()

            logger.log_epoch(ep, train_loss, val_err)
            logger.maybe_save_best(self.model, val_err)
            logger.save_latest_checkpoint(self.model, optimizer, scheduler, self.args, ep)

            if ep % 5 == 0:
                logger.save_periodic_checkpoint(self.model, optimizer, scheduler, self.args, ep)

        logger.save_final_model(self.model)
        logger.plot_curves()
        logger.close()

    def test(self):
        model_path = os.path.join('./experiments', self.args.save_name, 'best_model.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        result_dir = os.path.join('./experiments', self.args.save_name)
        os.makedirs(result_dir, exist_ok=True)

        log_path = os.path.join(result_dir, 'test_log.txt')
        log_file = open(log_path, 'w')
        print("Writing to:", log_path, flush=True)
        
        mean = torch.tensor(self.test_loader.dataset.coef_norm[2], dtype=torch.float32).cuda()
        std = torch.tensor(self.test_loader.dataset.coef_norm[3], dtype=torch.float32).cuda()

        all_metrics = []
        times = []
        metric_rows = []

        with torch.no_grad():
            for index, batch in enumerate(self.test_loader):
                if batch is None:
                    continue
                pos, fx, y, surf, geo = batch
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if geo.shape[0] == 1:
                    geo = geo.squeeze(0)
                if getattr(self.args, 'fun_dim', 0) == 0:
                    fx = None

                tic = time.time()
                out = self.model(x, fx, geo=geo)
                toc = time.time()

                #mean = torch.tensor(self.test_loader.dataset.coef_norm[2]).cuda()
                #std = torch.tensor(self.test_loader.dataset.coef_norm[3]).cuda()
                


                val_sets, val_n = [], []
                output_types = getattr(self.args, 'output_types', None)

                pred_pressC = out[..., -1][surf] * std[-1] + mean[-1]
                gt_pressC = y[..., -1][surf] * std[-1] + mean[-1]
                pred_velo = out[..., :-1][~surf] * std[:-2] + mean[:-2]
                gt_velo = y[..., :-2][~surf] * std[:-2] + mean[:-2]
                pred_spd = torch.norm(pred_velo, dim=-1)
                gt_spd = torch.norm(gt_velo, dim=-1) 
                
                mask = torch.norm(gt_velo, dim=-1) > 0.1
                if mask.sum() == 0:
                    print(f"Skipping index {index} due to empty velocity mask", flush=True)
                    continue
                pred_velo = pred_velo[mask]
                gt_velo = gt_velo[mask]
                
                val_sets.append((pred_pressC, gt_pressC))
                val_n.append('pres')
                val_sets.append((pred_velo, gt_velo))
                val_n.append('vel')
                val_sets.append((pred_spd, gt_spd))
                val_n.append('spd')

                geometry_name = self.dataset.split_ids['test'][index] if hasattr(self.dataset, 'split_ids') else f'sample_{index}'

                row = {'geometry': geometry_name, 'index': index}
                for (pred, gt), name in zip(val_sets, val_n):
                    metrics = compute_all_errors(pred, gt)
                    metrics_np = {f"{k}_{name}": (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in metrics.items()}
                    row.update(metrics_np)
                    all_metrics.append(metrics_np)
                metric_rows.append(row)

                times.append(toc - tic)

        df = pd.DataFrame(metric_rows)
        df.to_csv(os.path.join(result_dir, 'test_metrics.csv'), index=False)

        if len(all_metrics) == 0:
            log_file.write("No valid samples to compute aggregate metrics.\n")
            log_file.close()
            return

        if len(all_metrics) > 0:
            final_results = aggregate_metrics(metric_rows, apply_sqrt=False)
        else:
            print("No valid metrics computed. Skipping final aggregation.", flush=True)
            final_results = {}

        for k, v in final_results.items():
            if isinstance(v, np.ndarray):
                log_file.write(f"{k}: {v} (avg: {np.mean(v):.6f})\n")
            else:
                log_file.write(f"{k}: {v:.6f}\n")

        log_file.write(f"Inference Time (mean): {np.mean(times):.6f}s\n")
        log_file.close()
        print("Test results saved to:", log_path, flush=True)
        
    def test_all(self):
        model_path = os.path.join('./experiments', self.args.save_name, 'best_model.pt')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        result_dir = os.path.join('./experiments', self.args.save_name)
        os.makedirs(result_dir, exist_ok=True)
        
        test_types = ['val', 'test', 'real']
        test_loaders = [self.val_loader, self.test_loader, self.real_loader]
        
        for i in range(len(test_types)):

            log_path = os.path.join(result_dir, test_types[i]+'_log.txt')
            log_file = open(log_path, 'w')
            print("Writing to:", log_path, flush=True)

            all_metrics = []
            times = []
            metric_rows = []

            with torch.no_grad():
                for index, batch in enumerate(test_loaders[i]):
                    if batch is None:
                        continue
                    pos, fx, y, surf, geo = batch
                    x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                    if geo.shape[0] == 1:
                        geo = geo.squeeze(0)
                    if getattr(self.args, 'fun_dim', 0) == 0:
                        fx = None

                    tic = time.time()
                    out = self.model(x, fx, geo=geo)
                    toc = time.time()

                    #mean = torch.tensor(self.test_loader.dataset.coef_norm[2]).cuda()
                    #std = torch.tensor(self.test_loader.dataset.coef_norm[3]).cuda()

                    mean = torch.tensor(self.test_loader.dataset.coef_norm[2], dtype=torch.float32).cuda()
                    std = torch.tensor(self.test_loader.dataset.coef_norm[3], dtype=torch.float32).cuda()

                    val_sets, val_n = [], []
                    output_types = getattr(self.args, 'output_types', None)

                    pred_pressC = out[..., -1][surf] * std[-1] + mean[-1]
                    gt_pressC = y[..., -1][surf] * std[-1] + mean[-1]
                    pred_velo = out[..., :-1][~surf] * std[:-2] + mean[:-2]
                    gt_velo = y[..., :-2][~surf] * std[:-2] + mean[:-2]

                    mask = torch.norm(gt_velo, dim=-1) > 0.1
                    if mask.sum() == 0:
                        print(f"Skipping index {index} due to empty velocity mask", flush=True)
                        continue
                    pred_velo = pred_velo[mask]
                    gt_velo = gt_velo[mask]
                    
                    pred_spd = torch.norm(pred_velo, dim=-1)
                    gt_spd = torch.norm(gt_velo, dim=-1) 

                    val_sets.append((pred_pressC, gt_pressC))
                    val_n.append('pres')
                    val_sets.append((pred_velo, gt_velo))
                    val_n.append('vel')
                    val_sets.append((pred_spd, gt_spd))
                    val_n.append('spd')

                    geometry_name = self.dataset.split_ids[test_types[i]][index] if(hasattr(self.dataset, 'split_ids')and test_types[i]!='real') else self.dataset.real_ids if(test_types[i]!='real') else f'sample_{index}'

                    row = {'geometry': geometry_name, 'index': index}
                    for (pred, gt), name in zip(val_sets, val_n):
                        metrics = compute_all_errors(pred, gt)
                        metrics_np = {f"{k}_{name}": (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in metrics.items()}
                        row.update(metrics_np)
                        all_metrics.append(metrics_np)
                    metric_rows.append(row)

                    times.append(toc - tic)

            df = pd.DataFrame(metric_rows)
            df.to_csv(os.path.join(result_dir, test_types[i]+'_metrics.csv'), index=False)

            if len(all_metrics) == 0:
                log_file.write("No valid samples to compute aggregate metrics.\n")
                log_file.close()
                return

            if len(all_metrics) > 0:
                final_results = aggregate_metrics(metric_rows, apply_sqrt=False)
            else:
                print("No valid metrics computed. Skipping final aggregation.", flush=True)
                final_results = {}

            for k, v in final_results.items():
                if isinstance(v, np.ndarray):
                    log_file.write(f"{k}: {v} (avg: {np.mean(v):.6f})\n")
                else:
                    log_file.write(f"{k}: {v:.6f}\n")

            log_file.write(f"Inference Time (mean): {np.mean(times):.6f}s\n")
            log_file.close()
            print("Test results "+ test_types[i] + " saved to:", log_path, flush=True)
