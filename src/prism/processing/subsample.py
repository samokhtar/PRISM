# subsample.py
# Subsamples the mesh points 
# Licensed under the MIT License

from __future__ import annotations
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
import argparse
import os


def process_sample(root: str,
                   sample_rel: str,
                   bbox_min: np.ndarray,
                   bbox_max: np.ndarray,
                   target_N: int,
                   decay_xy: float,
                   decay_z: float,
                   suffix: str) -> None:
    try:
        parts = sample_rel.split(os.sep)
        if len(parts) != 2:
            return

        bname, child = parts
        _orient = int(child.split('_')[-1])

        parent = os.path.join(root, bname)
        child_dir = os.path.join(parent, child)

        path_x = os.path.join(parent, 'x.npy')
        path_srf = os.path.join(parent, 'surf.npy')
        path_pos = os.path.join(parent, 'pos.npy')
        path_y = os.path.join(child_dir, 'y.npy')
        path_xrot = os.path.join(child_dir, 'x_rot.npy')
        path_posrot = os.path.join(child_dir, 'pos_rot.npy')

        paths_all = [path_srf, path_x, path_pos, path_xrot, path_posrot, path_y]

        if not all(os.path.exists(p) for p in paths_all):
            return

        surf = np.load(path_srf)             
        pos = np.load(path_pos)             
        y = np.load(path_y)                    

        surface_mask = (surf == 1)
        nonsurface_mask = (surf == 0)

        pos_surface = pos[surface_mask]
        pos_nonsurface = pos[nonsurface_mask]
        y_nonsurface = y[nonsurface_mask]

        inside_mask = np.all((pos_nonsurface >= bbox_min) & (pos_nonsurface <= bbox_max), axis=1)
        y_valid_mask = ~np.all(np.isclose(y_nonsurface, 0.0), axis=-1)
        combined_mask = inside_mask & y_valid_mask

        coords_filtered = pos_nonsurface[combined_mask]
        if coords_filtered.shape[0] == 0:
            print(f"No valid nonsurface points in {sample_rel}", flush=True)
            return

        dist_xy = np.linalg.norm(coords_filtered[:, :2], axis=1)
        z_penalty = np.exp(-decay_z * coords_filtered[:, 2])
        scores = np.exp(-decay_xy * dist_xy) * z_penalty
        scores += np.random.uniform(0, 1e-6, size=scores.shape)

        n_surface = pos_surface.shape[0]
        n_remaining = max(int(target_N) - n_surface, 0)

        if coords_filtered.shape[0] <= n_remaining:
            top_k_indices = np.arange(coords_filtered.shape[0])
        else:
            top_k_indices = np.argpartition(-scores, n_remaining)[:n_remaining]

        for p in paths_all:
            arr = np.load(p)

            arr_surface = arr[surface_mask]
            arr_nonsurface = arr[nonsurface_mask]
            arr_nonsurface_filtered = arr_nonsurface[combined_mask]
            arr_final = np.concatenate([arr_surface, arr_nonsurface_filtered[top_k_indices]], axis=0)

            out_path = p.replace('.npy', f'_{suffix}.npy')
            np.save(out_path, arr_final)

        print(f"Processed {sample_rel} with {n_surface} surface + {len(top_k_indices)} sampled", flush=True)

    except Exception as e:
        print(f"Error processing {sample_rel}: {e}", flush=True)


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
    ap = argparse.ArgumentParser(description="Downselect to target_N points: keep all surface + sampled nonsurface.")
    ap.add_argument("--out_dir", required=True, help="Root directory of vtknpy outputs.")
    ap.add_argument("--target_N", type=int, default=1_000_000, help="Total desired points per sample (default: 1,000,000).")
    ap.add_argument("--bbox_min", type=float, nargs=3, default=[-550, -550, 0], help="Min XYZ for bbox filter.")
    ap.add_argument("--bbox_max", type=float, nargs=3, default=[550, 550, 1100], help="Max XYZ for bbox filter.")
    ap.add_argument("--decay_xy", type=float, default=0.2, help="Decay factor in XY distance.")
    ap.add_argument("--decay_z", type=float, default=0.2, help="Decay factor in Z.")
    ap.add_argument("--suffix", default="1M", help="Suffix for saved arrays (default: '1M').")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers (joblib).")
    ap.add_argument("--backend", default="multiprocessing",
                    choices=["loky", "threading", "multiprocessing"],
                    help="Joblib backend (default: multiprocessing).")
    args = ap.parse_args()

    root = Path(args.out_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    samples = discover_samples(root)
    if len(samples) == 0:
        print("No samples found (expected <bname>/<bname>_<orient> folders).")
        return

    bbox_min = np.array(args.bbox_min, dtype=np.float32)
    bbox_max = np.array(args.bbox_max, dtype=np.float32)

    Parallel(n_jobs=args.jobs, backend=args.backend)(
        delayed(process_sample)(
            str(root),
            s,
            bbox_min,
            bbox_max,
            args.target_N,
            args.decay_xy,
            args.decay_z,
            args.suffix
        )
        for s in samples
    )


if __name__ == "__main__":
    main()
