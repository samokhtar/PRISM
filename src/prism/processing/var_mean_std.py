# dataset_mean_std.py
# Calculates the mean and standard deviation of variables for normalization
# Licensed under the MIT License

from __future__ import annotations
import os
import argparse
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np


def process_mean(root: str, sample_rel: str, velocity_threshold: float):
    try:
        base = os.path.join(root, '_'.join(sample_rel.split('_')[0:2]), sample_rel)

        path_x = os.path.join(base, 'x_rot.npy')
        path_y = os.path.join(base, 'y.npy')
        if not (os.path.exists(path_x) and os.path.exists(path_y)):
            return None

        x = np.load(path_x)
        y = np.load(path_y)

        U = y[:, :3]
        magU = np.linalg.norm(U, axis=1)
        y[magU > velocity_threshold, :3] = 0

        xy = np.concatenate([x, y], axis=1)
        valid_mask = ~np.isnan(xy).any(axis=1)

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        return x_valid.sum(axis=0), y_valid.sum(axis=0), x_valid.shape[0]

    except Exception as e:
        print(f"Error in mean pass on {sample_rel}: {e}")
        return None


def process_var(root: str, sample_rel: str, mean_x: np.ndarray, mean_y: np.ndarray, velocity_threshold: float):
    try:
        base = os.path.join(root, '_'.join(sample_rel.split('_')[0:2]), sample_rel)

        path_x = os.path.join(base, 'x_rot.npy')
        path_y = os.path.join(base, 'y.npy')
        if not (os.path.exists(path_x) and os.path.exists(path_y)):
            return None

        x = np.load(path_x)
        y = np.load(path_y)

        U = y[:, :3]
        magU = np.linalg.norm(U, axis=1)
        y[magU > velocity_threshold, :3] = 0

        xy = np.concatenate([x, y], axis=1)
        valid_mask = ~np.isnan(xy).any(axis=1)

        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        sq_dev_x = ((x_valid - mean_x) ** 2).sum(axis=0)
        sq_dev_y = ((y_valid - mean_y) ** 2).sum(axis=0)

        return sq_dev_x, sq_dev_y

    except Exception as e:
        print(f"Error in variance pass on {sample_rel}: {e}")
        return None


def discover_samples(out_dir: Path) -> list[str]:
    samples = []
    for child in out_dir.rglob('*'):
        if not child.is_dir():
            continue
        rel = child.relative_to(out_dir)
        if len(rel.parts) != 2:
            continue
        bname, childname = rel.parts
        if childname.startswith(bname + '_'):
            samples.append(str(rel))
    return samples


def main():
    ap = argparse.ArgumentParser(description="Two-pass reduction to compute mean/std for x_rot and y across samples.")
    ap.add_argument("--out_dir", required=True, help="Root directory (contains <bname>/<bname>_<orient>).")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers.")
    ap.add_argument("--backend", default="multiprocessing",
                    choices=["loky", "threading", "multiprocessing"],
                    help="Joblib backend.")
    ap.add_argument("--velocity_threshold", type=float, default=100.0,
                    help="Zero out Ux,Uy,Uz when ||U|| exceeds this threshold.")
    args = ap.parse_args()

    root_path = Path(args.out_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"{root_path} does not exist")

    samples = discover_samples(root_path)
    if len(samples) == 0:
        print("No samples found (expected <bname>/<bname>_<orient> folders).")
        return

    root_str = str(root_path)

    results = Parallel(n_jobs=args.jobs, backend=args.backend)(
        delayed(process_mean)(root_str, s, args.velocity_threshold) for s in samples
    )

    total_sum_x, total_sum_y, total_count = 0, 0, 0
    for r in results:
        if r is not None:
            sx, sy, count = r
            total_sum_x += sx
            total_sum_y += sy
            total_count += count

    if total_count == 0:
        raise RuntimeError("No valid rows found across samples (total_count == 0).")

    mean_x = total_sum_x / total_count
    mean_y = total_sum_y / total_count

    np.save(root_path / 'mean_x.npy', mean_x)
    np.save(root_path / 'mean_y.npy', mean_y)
    np.save(root_path / 'total_count.npy', np.array(total_count, dtype=np.int64))

    print("Mean x:", mean_x)
    print("Mean y:", mean_y)

    var_results = Parallel(n_jobs=args.jobs, backend=args.backend)(
        delayed(process_var)(root_str, s, mean_x, mean_y, args.velocity_threshold) for s in samples
    )

    total_sq_dev_x, total_sq_dev_y = 0, 0
    for r in var_results:
        if r is not None:
            dx, dy = r
            total_sq_dev_x += dx
            total_sq_dev_y += dy

    var_x = total_sq_dev_x / total_count
    var_y = total_sq_dev_y / total_count
    std_x = np.sqrt(var_x)
    std_y = np.sqrt(var_y)

    print("Std x:", std_x)
    print("Std y:", std_y)

    np.save(root_path / 'coef_norm.npy', [mean_x, std_x, mean_y, std_y])


if __name__ == "__main__":
    main()
