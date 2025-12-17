# shape_index_ext.py
# Licensed under the MIT License

# Parts of this code have been adapted from the following sources: 

# https://github.com/tudelft3d/3d-building-metrics, 3d-building-metrics/shape_index.py
# Modified: adapted to OBJ shapes and without pymesh
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


import math
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon
import miniball
import numpy as np
import pyvista as pv
from polylabel import polylabel

from 3d_building_metrics.shape_index import *
from 3d_building_metrics.helpers.smallestenclosingcircle import make_circle
from 3d_building_metrics.helpers.geometry import surface_normal

def create_surface_grid(mesh, density=1):

    result = []

    sized = mesh.compute_cell_sizes()

    faces = mesh.faces
    points = mesh.points

    offset = 0
    for i in range(mesh.n_cells):

        n_pts = faces[offset]
        ids = faces[offset + 1 : offset + 1 + n_pts]
        offset += n_pts + 1

        pts = points[ids]

        try:
            normal = surface_normal(pts)
        except Exception:
            continue

        pts_2d = project_2d(pts, normal)
        poly_2d = Polygon(pts_2d)

        if not poly_2d.is_valid:
            continue

        grid = create_grid_2d(poly_2d, density)
        grid = MultiPoint(grid).intersection(poly_2d)

        if grid.is_empty:
            continue
        elif grid.geom_type == "Point":
            grid = np.array(grid.coords)
        else:
            grid = np.array([list(p.coords[0]) for p in grid.geoms])

        result.extend(list(to_3d(grid, normal, pts[0])))

    return result

def exchange_3d(mesh, evs=None, density=0.25, engine="igl"):

    if evs is None:
        voxel = pv.voxelize(mesh, density=density, check_surface=False)
        grid = voxel.cell_centers().points

        if len(grid) == 0:
            centroid = mesh.center
        else:
            centroid = np.mean(grid, axis=0)

        evs = equal_volume_sphere(mesh, centroid)
    
    if mesh.n_open_edges > 0:
        return -1

    try:
        inter = mesh.boolean_intersection(evs)
    except:
        return -1

    return inter.volume / mesh.volume

def spin_2d(shape, grid=None, density=1):
    if grid is None:
        grid = create_grid_2d(shape, density)
    
    if isinstance(grid, list):
        grid = MultiPoint(grid).intersection(shape)

        if grid.is_empty:
            return -1

        if grid.geom_type == "Point":
            grid = MultiPoint([grid])

    centroid = shape.centroid

    return 0.5 * (shape.area / math.pi) / np.mean([math.pow(centroid.distance(p), 2) for p in grid.geoms])

def spin_3d(mesh, grid=None, density=1, check_surface=False):
    if grid is None:
        voxel = pv.voxelize(mesh, density=density, check_surface=check_surface)
        grid = voxel.cell_centers().points
    
    centroid = np.mean(grid, axis=0)
    
    r = math.pow(3 * mesh.volume / (4 * math.pi), 1/3)
    return 3 / 5 * math.pow(r, 2) / np.mean([math.pow(distance(centroid, p), 2) for p in grid])

def depth_2d(shape, grid=None, density=1):
    if grid is None:
        grid = create_grid_2d(shape, density)
    
    if isinstance(grid, list):
        grid = MultiPoint(grid).intersection(shape)

        if grid.is_empty:
            return -1

        if grid.geom_type == "Point":
            grid = MultiPoint([grid])
        
    return 3 * np.mean([p.distance(shape.boundary) for p in grid.geoms]) / math.sqrt(shape.area / math.pi)

def largest_inscribed_circle(shape):

    def _lic_from_polygon(polygon):

        rings = [list(polygon.exterior.coords)]
        rings += [list(ring.coords) for ring in polygon.interiors]
        center, r = polylabel(rings, with_distance=True)
        return Point(center).buffer(r), r

    if isinstance(shape, Polygon):
        lic, _ = _lic_from_polygon(shape)
        return lic

    elif isinstance(shape, MultiPolygon):
        largest_circle = None
        max_radius = -1

        for geom in shape.geoms:
            lic, r = _lic_from_polygon(geom)
            if r > max_radius:
                max_radius = r
                largest_circle = lic

        return largest_circle

    else:
        raise ValueError(f"Unsupported geometry type: {shape.geom_type}")

def range_2d(shape):

    if isinstance(shape, MultiPolygon):
        coords = [pt for geom in shape.geoms for pt in geom.exterior.coords]
    elif isinstance(shape, Polygon):
        coords = list(shape.exterior.coords)
    else:
        raise ValueError(f"Unsupported geometry type: {shape.geom_type}")

    x, y, r = make_circle([c[:2] for c in coords])

    return math.sqrt(shape.area / math.pi) / r

