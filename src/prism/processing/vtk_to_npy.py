# vtk_to_npy.py
# Orchestrates: per-building surface pack (pos/surf/x) and per-orientation field pack (y)
# Licensed under the MIT License

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

from __future__ import annotations
from typing import Optional, Tuple, Dict
import argparse
import numpy as np
import traceback
from multiprocessing import Pool, cpu_count
from prism.processing.geom_utils import (
    load_poly_data, load_unstructured_grid_data, get_point_normals_from_any,
    get_edges, load_trimesh_obj_as_upY, signed_distance_and_normals, npy_exists_all
)
from vtk.util.numpy_support import vtk_to_numpy


def find_first_available_orientation(vtk_dir: Path, bname: str, orientations: Tuple[int, ...]) -> Optional[int]:
    for orient in orientations:
        u = vtk_dir / bname / f"{bname}_{orient}_U.vtk"
        p = vtk_dir / bname / f"{bname}_{orient}_P.vtk"
        if u.exists() and p.exists():
            return orient
    return None


def save_surface_pack(
    bname: str,
    vtk_dir: Path,
    geo_dir: Path,
    out_dir: Path,
    orient: int,
    sdf_mode="igl"
) -> bool:

    save_path = out_dir / bname
    save_path.mkdir(parents=True, exist_ok=True)

    fU = vtk_dir / bname / f"{bname}_{orient}_U.vtk"
    fP = vtk_dir / bname / f"{bname}_{orient}_P.vtk"
    obj = geo_dir / f"{bname}.obj"

    if npy_exists_all(save_path, ["x", "pos", "surf"]):
        return True

    poly_P = load_poly_data(fP)
    pts_P = vtk_to_numpy(poly_P.GetPoints().GetData()).astype(np.float32)
    np.save(save_path / "pts_P.npy", pts_P)

    edges_P = get_edges(poly_P, pts_P)
    np.save(save_path / "edges_P.npy", edges_P)

    normal_P = get_point_normals_from_any(poly_P).astype(np.float32)
    sdf_P = np.zeros(pts_P.shape[0], dtype=np.float32)  # by definition on surface


    unstr_U = load_unstructured_grid_data(fU)
    pts_U = vtk_to_numpy(unstr_U.GetPoints().GetData()).astype(np.float32)
    np.save(save_path / "pts_U.npy", pts_U)

    surf_mesh = load_trimesh_obj_as_upY(obj)
    sdf_U, normal_U = compute_sdf_from_mesh(pts_U, srf_mesh, method=sdf_mode)

    surface_set = {tuple(p) for p in pts_P}
    ext_idx = [i for i, p in enumerate(pts_U) if tuple(p) not in surface_set]

    pos_ext = pts_U[ext_idx]
    pos_srf = pts_P
    pos = np.concatenate([pos_ext, pos_srf], axis=0)
    np.save(save_path / "pos.npy", pos)

    surf = np.concatenate([np.zeros(len(pos_ext), dtype=np.float32),
                           np.ones(len(pos_srf), dtype=np.float32)], axis=0)
    np.save(save_path / "surf.npy", surf)

    sdf_ext = sdf_U[ext_idx]
    sdf_srf = sdf_P
    nrm_ext = normals_U[ext_idx]
    nrm_srf = normal_P

    init_ext = np.c_[pos_ext, sdf_ext, nrm_ext]
    init_srf = np.c_[pos_srf, sdf_srf, nrm_srf]
    init = np.concatenate([init_ext, init_srf], axis=0).astype(np.float32)

    np.save(save_path / "x.npy", init)
    return True


def save_orientation_pack(
    bname: str,
    orient: int,
    vtk_dir: Path,
    out_dir: Path
) -> bool:

    b_out = out_dir / bname
    if not npy_exists_all(b_out, ["x", "pos", "surf", "pts_P", "pts_U"]):
        needed = ["x", "pos", "surf", "pts_P", "pts_U"]
        missing = [n for n in needed if not (b_out / f"{n}.npy").exists()]
        raise FileNotFoundError(f"Missing surface pack for {bname}: {missing}")

    fU = vtk_dir / bname / f"{bname}_{orient}_U.vtk"
    fP = vtk_dir / bname / f"{bname}_{orient}_P.vtk"
    if not (fU.exists() and fP.exists()):
        return False

    save_path = out_dir / bname / f"{bname}_{orient}"
    save_path.mkdir(parents=True, exist_ok=True)
    y_path = save_path / "y.npy"
    if y_path.exists():
        return True 

    poly_P = load_poly_data(fP)
    pts_P = np.load(b_out / "pts_P.npy")
    P = vtk_to_numpy(poly_P.GetPointData().GetArray("p")).astype(np.float32)
    P_c = vtk_to_numpy(poly_P.GetPointData().GetArray("total(p)_coeff")).astype(np.float32)

    unstr_U = load_unstructured_grid_data(fU)
    pts_U = np.load(b_out / "pts_U.npy")
    U = vtk_to_numpy(unstr_U.GetPointData().GetArray("U")).astype(np.float32)

    surface_set = {tuple(p) for p in pts_P}
    ext_idx = [i for i, p in enumerate(pts_U) if tuple(p) not in surface_set]

    U_dict: Dict[tuple, np.ndarray] = {tuple(p): U[i] for i, p in enumerate(pts_U)}
    U_surf = np.array([U_dict.get(tuple(p), np.zeros(3, dtype=np.float32)) for p in pts_P], dtype=np.float32)

    U_ext = U[ext_idx]
    P_ext = np.zeros((len(ext_idx), 2), dtype=np.float32)
    target_ext = np.c_[U_ext, P_ext].astype(np.float32)

    P_surf = np.stack([P, P_c], axis=1)  # (N,2)
    target_surf = np.c_[U_surf, P_surf].astype(np.float32)

    target = np.concatenate([target_ext, target_surf], axis=0)
    np.save(y_path, target)
    return True


def process_one_building(args) -> Tuple[str, bool]:
    (bname, vtk_dir, geo_dir, out_dir, orientations) = args
    try:
        first = find_first_available_orientation(vtk_dir, bname, orientations)
        if first is None:
            print(f"[skip] {bname}: no valid orientations found")
            return bname, False

        ok_surface = save_surface_pack(vtk_dir, out_dir, geo_dir, b_name, available_orient, sdf_mode)
        if not ok_surface:
            print(f"[fail] {bname}: surface pack failed")
            return bname, False

        for o in orientations:
            try:
                save_orientation_pack(bname, o, vtk_dir, out_dir)
            except Exception as e:
                print(f"[warn] {bname} orient {o}: {e}")
                continue

        return bname, True

    except Exception as e:
        print(f"[error] {bname}: {e}")
        traceback.print_exc()
        return bname, False


def main():
    parser = argparse.ArgumentParser(description="Convert VTK CFD outputs + OBJ geo to NPY packs.")
    parser.add_argument("--vtk_dir", type=Path, required=True, help="Directory with <bname>/<bname>_<orient>_{U,P}.vtk")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to write npy packs")
    parser.add_argument("--geo_dir", type=Path, required=True, help="Directory with <bname>.obj geometry")
    parser.add_argument("--jobs", type=int, default=max(1, cpu_count() - 1), help="Parallel workers")
    parser.add_argument("--orient_step", type=int, default=45, help="Orientation step in degrees")
    parser.add_argument("--orient_count", type=int, default=8, help="Number of orientations")
    parser.add_argument("--sdf_mode", default="pc_unsigned", choices=["pc_unsigned", "mesh_signed"],help="How to compute SDF for U points (default: pc_unsigned).")
    args = parser.parse_args()

    vtk_dir: Path = args.vtk_dir
    out_dir: Path = args.out_dir
    geo_dir: Path = args.geo_dir
    sdf_mode = args.sdf_mode
    jobs: int = args.jobs

    orientations = tuple(int(i * args.orient_step) for i in range(args.orient_count))

    bnames = sorted([p.name for p in vtk_dir.iterdir() if p.is_dir()])

    tasks = [(b, vtk_dir, geo_dir, out_dir, orientations) for b in bnames]
    if jobs <= 1:
        results = [process_one_building(t) for t in tasks]
    else:
        with Pool(processes=jobs) as pool:
            results = pool.map(process_one_building, tasks, chunksize=1)

    ok = sum(1 for _, s in results if s)
    print(f"\nDone. {ok}/{len(results)} buildings processed successfully.")


if __name__ == "__main__":
    main()
