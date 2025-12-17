# convert_vtk_to_npy.py
# Orchestrates: per-building surface pack (pos/surf/x) and per-orientation field pack (y)
# Licensed under the MIT License

from __future__ import annotations
from joblib import Parallel, delayed
import numpy as np
from pathlib import Path
import argparse
import os

def rotate_points_and_normals_np(xyz: np.ndarray,
                                 normals: np.ndarray,
                                 R: np.ndarray,
                                 center: np.ndarray | None = None,
                                 normalize_normals: bool = True) -> tuple[np.ndarray, np.ndarray]:

    assert xyz.shape[1] == 3
    assert normals.shape[1] == 3
    assert R.shape == (3, 3)

    if center is not None:
        xyz_centered = xyz - center
    else:
        xyz_centered = xyz

    xyz_rot = xyz_centered @ R.T
    if center is not None:
        xyz_rot = xyz_rot + center

    normals_rot = np.empty_like(normals)
    normal_lengths = np.linalg.norm(normals, axis=-1)
    valid_normals = normal_lengths > 1e-6
    normals_rot[valid_normals] = normals[valid_normals] @ R.T
    normals_rot[~valid_normals] = normals[~valid_normals]

    if normalize_normals:
        norms = np.linalg.norm(normals_rot, axis=-1, keepdims=True) + 1e-8
        normals_rot = normals_rot / norms

    return xyz_rot, normals_rot


def np_rotation_matrix_z(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def process_sample(root_dir: str, sample_rel: str) -> None:
    try:
        parts = sample_rel.split(os.sep)
        if len(parts) != 2:
            return
        bname, child = parts
        toks = child.split('_')
        orient = int(toks[-1]) 

        parent = os.path.join(root_dir, bname)
        child_dir = os.path.join(parent, child)

        path_x = os.path.join(parent, 'x.npy')
        path_y = os.path.join(child_dir, 'y.npy')
        path_xrot = os.path.join(child_dir, 'x_rot.npy')
        path_posrot = os.path.join(child_dir, 'pos_rot.npy')
        path_stats_x = os.path.join(child_dir, 'stats_x.npy')
        path_stats_y = os.path.join(child_dir, 'stats_y.npy')
        path_stats_ct = os.path.join(child_dir, 'stats_ct.npy')

        if not (os.path.exists(path_x) and os.path.exists(path_y)):
            return

        if all(os.path.exists(p) for p in [path_xrot, path_posrot, path_stats_x, path_stats_y, path_stats_ct]):
            return

        x = np.load(path_x) 
        y = np.load(path_y)

        xyz = x[:, :3]
        normals = x[:, 4:7] if x.shape[1] >= 7 else np.zeros_like(xyz)

        R = np_rotation_matrix_z(orient)
        center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        xyz_rot, normals_rot = rotate_points_and_normals_np(xyz, normals, R, center=center)
        x_rot = x.copy()
        x_rot[:, :3] = xyz_rot
        if x.shape[1] >= 7:
            x_rot[:, 4:7] = normals_rot

        np.save(path_xrot, x_rot)
        np.save(path_posrot, xyz_rot)

        xy = np.concatenate([x_rot, y], axis=1)
        valid_mask = ~np.isnan(xy).any(axis=1)

        x_valid = x_rot[valid_mask]
        y_valid = y[valid_mask]

        np.save(path_stats_x, x_valid.sum(axis=0))
        np.save(path_stats_y, y_valid.sum(axis=0))
        np.save(path_stats_ct, np.array(x_valid.shape[0], dtype=np.int64))

        print(f"Processed {sample_rel}")

    except Exception as e:
        print(f"Error processing {sample_rel}: {e}")


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
    ap = argparse.ArgumentParser(description="Rotate x/pos and compute stats for each orientation folder.")
    ap.add_argument("--out_dir", required=True, help="Root directory of vtknpy outputs (parent has x.npy; children have y.npy).")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel workers (joblib).")
    ap.add_argument("--backend", default="multiprocessing", choices=["loky", "threading", "multiprocessing"],
                    help="Joblib backend (default: multiprocessing).")
    args = ap.parse_args()

    root = Path(args.out_dir)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist")

    samples = discover_samples(root)
    if len(samples) == 0:
        print("No samples found (expected <bname>/<bname>_<orient> folders).")
        return

    Parallel(n_jobs=args.jobs, backend=args.backend)(
        delayed(process_sample)(str(root), s) for s in samples
    )


if __name__ == "__main__":
    main()
