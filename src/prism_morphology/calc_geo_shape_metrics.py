# calc_geo_shape_metrics.py
# Licensed under the MIT License

# Parts of this code have been adapted from the following sources: 

# https://github.com/tudelft3d/3d-building-metrics
# Licensed under the MIT License
# © Original Author(s): 3D geoinformation research group at TU Delft

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
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VTK_SMP_MAX_THREADS"] = "1"

import sys
import argparse
import numpy as np
import trimesh
import pyvista as pv
import scipy.spatial as ss
from pathlib import Path
from shapely.ops import unary_union
from multiprocessing import Pool

from shapely.geometry import Polygon, MultiPolygon  
from prsim_morphology.shape_index_ext as si  
from 3d_building_metrics.helpers.geometry import to_shapely_from_pyvista, boundingbox_volume 
from 3d_building_metrics import geometry  


def area_by_orientation(mesh_trimesh, horizontal_threshold=0.9):
    normals = mesh_trimesh.face_normals
    cos_theta = np.abs(normals @ np.array([0, 0, 1]))
    horizontal_mask = cos_theta >= horizontal_threshold
    horizontal_area = mesh_trimesh.area_faces[horizontal_mask].sum()
    other_area = mesh_trimesh.area_faces[~horizontal_mask].sum()
    return {
        'horizontal_area': float(horizontal_area),
        'other_area': float(other_area),
        'total_area': float(horizontal_area + other_area),
    }

def trimesh_load_any(path: str) -> trimesh.Trimesh:
    scene = trimesh.load(path, force='scene')
    if isinstance(scene, trimesh.Scene):
        geoms = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            m = trimesh.load(path)
            if isinstance(m, trimesh.Trimesh):
                return m
            raise ValueError(f"No Trimesh geometry found in {path}")
        return trimesh.util.concatenate(geoms)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    raise ValueError(f"Unsupported trimesh type for {path}: {type(scene)}")

def derive_metrics_root(geo_dir: Path) -> Path:
    s = str(geo_dir)
    if "/geometry/" in s:
        return Path(s.replace("/geometry/", "/metrics/"))
    if s.endswith("/geometry"):
        return Path(s[:-9] + "metrics")
    return geo_dir.parent / (geo_dir.name + "_metrics")

def discover_items(geo_dir: Path, metrics_root: Path):
    """
    Walk geo_dir for *.obj and produce items with:
      - path     : full input OBJ path
      - out_dir  : output directory (metrics mirror + parent folder)
      - prefix   : file prefix (parent2_parent1 or parent1)
      - label    : short display label
    """
    for f in geo_dir.rglob("*.obj"):
        rel = f.relative_to(geo_dir)
        parents = list(rel.parents)
        parent1 = f.parent.name
        parent2 = f.parent.parent.name if f.parent.parent != geo_dir and f.parent.parent != f.parent else None

        mirror_rel = rel.parent
        out_dir = metrics_root / mirror_rel / ""  
        out_dir = out_dir / ""  

        # sane prefix
        prefix = f"{parent2}_{parent1}" if parent2 else parent1
        label = str(rel)

        yield {
            "path": str(f),
            "out_dir": str(out_dir),
            "prefix": prefix,
            "label": label,
        }

def process_one(item, args):
    try:
        geo_path = item["path"]
        out_dir = item["out_dir"]
        prefix = item["prefix"]
        label = item["label"]

        if not os.path.exists(geo_path) or os.path.getsize(geo_path) < 100:
            print(f"[skip] {label}: missing/tiny file", flush=True)
            return

        if os.path.exists(out_dir) and len(os.listdir(out_dir)) == 3 and not bool(args.override):
            print(f"[skip] {label}: already done", flush=True)
            return

        mesh_pv = pv.read(geo_path)
        mesh_tri = trimesh_load_any(geo_path)
        print(f"[load] {label}: {mesh_tri.vertices.shape[0]} vertices", flush=True)

        if bool(getattr(args, 'swap_yz', 0)):
            mesh_pv.points[:, [1, 2]] = mesh_pv.points[:, [2, 1]]
            mesh_tri.vertices[:, [1, 2]] = mesh_tri.vertices[:, [2, 1]]

        points = mesh_tri.vertices
        convex_hull = ss.ConvexHull(points)

        shapely_geom = to_shapely_from_pyvista(mesh_pv, ground_only=False)
        if isinstance(shapely_geom, list):
            shapely_geom = unary_union(shapely_geom)

        obb_2d = shapely_geom.minimum_rotated_rectangle
        min_z = float(np.min(mesh_tri.vertices[:, 2]))
        max_z = float(np.max(mesh_tri.vertices[:, 2]))
        obb = geometry.extrude(obb_2d, min_z, max_z)

        s_area = area_by_orientation(mesh_tri, horizontal_threshold=args.horz_t)

        actual_volume = float(abs(mesh_tri.volume))
        convexhull_volume = float(abs(convex_hull.volume))
        obb_volume = float(abs(obb.volume))
        aabb_volume = float(boundingbox_volume(points))
        footprint_perimeter = float(abs(shapely_geom.length))
        footprint_area = float(abs(shapely_geom.area))
        total_area = float(abs(s_area['total_area']))
        vert_area = float(abs(s_area['other_area']))
        horz_area = float(abs(s_area['horizontal_area']))
        S, L = si.get_box_dimensions(obb_2d)

        voxel = pv.voxelize(mesh_pv, density=args.density_3d, check_surface=False)
        grid = voxel.cell_centers().points
        tri_mesh = mesh_pv.triangulate().clean()

        shape_2d = shapely_geom
        shape_3d = mesh_pv
        shape_3d_tri = tri_mesh

        circularity_2d = si.circularity(shape_2d)
        hemisphericality_3d = si.hemisphericality(shape_3d)
        convexity_2d = shape_2d.area / convex_hull.area
        convexity_3d = shape_3d.volume / convex_hull.volume
        fractality_2d = si.fractality_2d(shape_2d)
        fractality_3d = si.fractality_3d(shape_3d)
        rectangularity_2d = shape_2d.area / shape_2d.minimum_rotated_rectangle.area
        rectangularity_3d = shape_3d.volume / obb.volume
        squareness_2d = si.squareness(shape_2d)
        cubeness_3d = si.cubeness(shape_3d)
        horizontal_elongation = si.elongation(S, L)
        min_vertical_elongation = si.elongation(L, max_z)
        max_vertical_elongation = si.elongation(S, max_z)
        form_factor_3D = shape_2d.area / (shape_3d.volume**(2/3))
        equivalent_rectangularity_index_2d = si.equivalent_rectangular_index(shape_2d)
        equivalent_prism_index_3d = si.equivalent_prism_index(shape_3d, obb)
        proximity_index_2d = si.proximity_2d(shape_2d, density=args.density_2d)
        proximity_index_3d = si.proximity_3d(shape_3d_tri, grid, density=args.density_3d) if len(grid) > 2 else "NA"
        exchange_index_2d = si.exchange_2d(shape_2d)
        exchange_index_3d = si.exchange_3d(shape_3d_tri, density=args.density_3d)
        spin_index_2d = si.spin_2d(shape_2d, density=args.density_2d)
        spin_index_3d = si.spin_3d(shape_3d_tri, grid, density=args.density_3d) if len(grid) > 2 else "NA"
        perimeter_index_2d = si.perimeter_index(shape_2d)
        circumference_index_3d = si.circumference_index_3d(shape_3d_tri)
        depth_index_2d = si.depth_2d(shape_2d, density=args.density_2d)
        depth_index_3d = si.depth_3d(shape_3d_tri, density=args.density_3d) if len(grid) > 2 else "NA"
        roughness_index_2d = si.roughness_index_2d(shape_2d, density=args.density_2d)
        roughness_index_3d = si.roughness_index_3d(shape_3d_tri, grid, args.density_2d) if len(grid) > 2 else "NA"
        girth_index_2d = si.girth_2d(shape_2d)
        girth_index_3d = si.girth_3d(tri_mesh, grid, density=args.density_3d) if len(grid) > 2 else "NA"
        dispersion_index_2d = si.dispersion_2d(shape_2d, density=args.density_2d)
        dispersion_index_3d = si.dispersion_3d(shape_3d, grid, density=args.density_3d) if len(grid) > 2 else "NA"
        range_index_2d = si.range_2d(shape_2d)
        range_index_3d = si.range_3d(shape_3d)

        params_2d = [
            circularity_2d, convexity_2d, fractality_2d, rectangularity_2d, squareness_2d,
            equivalent_rectangularity_index_2d, proximity_index_2d, exchange_index_2d,
            spin_index_2d, perimeter_index_2d, depth_index_2d, roughness_index_2d,
            girth_index_2d, dispersion_index_2d, range_index_2d,
        ]

        params_3d = [
            hemisphericality_3d, convexity_3d, fractality_3d, rectangularity_3d, cubeness_3d, form_factor_3D,
            equivalent_prism_index_3d, proximity_index_3d, exchange_index_3d, spin_index_3d, circumference_index_3d,
            depth_index_3d, roughness_index_3d, girth_index_3d, dispersion_index_3d, range_index_3d,
        ]

        params_properties = [
            actual_volume, convexhull_volume, obb_volume, aabb_volume, footprint_perimeter,
            footprint_area, total_area, vert_area, horz_area, max_z, L, S,
            horizontal_elongation, min_vertical_elongation, max_vertical_elongation,
        ]

        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{prefix}_2dmetrics.npy"), params_2d)
        np.save(os.path.join(out_dir, f"{prefix}_3dmetrics.npy"), params_3d)
        np.save(os.path.join(out_dir, f"{prefix}_properties.npy"), params_properties)

        print(f"[ok] {label}", flush=True)

    except Exception as e:
        print(f"[fail] {label} :: {e}", flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geo_dir", type=str, required=True, help="Root folder containing OBJ files anywhere below.")
    p.add_argument("--horz_t", type=float, default=0.95)
    p.add_argument("--density_3d", type=float, default=1.0)
    p.add_argument("--density_2d", type=float, default=1.0)
    p.add_argument("--override", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--swap_yz", type=int, default=0, help="Swap Y/Z axes before metrics if needed.")
    args = p.parse_args()

    geo_dir = Path(args.geo_dir).resolve()
    metrics_root = derive_metrics_root(geo_dir)

    items = list(discover_items(geo_dir, metrics_root))
    print(f"Discovered {len(items)} OBJ files under {geo_dir}", flush=True)
    if not items:
        return

    with Pool(processes=args.workers) as pool:
        pool.starmap(process_one, [(item, args) for item in items])

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
