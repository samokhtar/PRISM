# geom_utils.py
# Utilities for loading VTK/mesh files and computing normals, edges, SDF, etc.
# Licensed under the MIT License

# Parts of this code have been adapted from the following sources: 

# https://github.com/thuml/Neural-Solver-Library, Neural-Solver-Library/data_provider/shapenet_utils.py
# Modified: adapted to data structure and outputs, SDF calculated directly from geometry
# Licensed under the MIT License
# © Original Author(s): Haixu Wu and Huakun Luo and Haowen Wang and Jianmin Wang and Mingsheng Long

import torch
import vtk
import os
import itertools
import random
import numpy as np
import igl
import trimesh
import torch_geometric
from torch_geometric import nn as nng
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import k_hop_subgraph, subgraph
from vtk.util.numpy_support import vtk_to_numpy
from pathlib import Path
import gc
import psutil

def load_unstructured_grid_data(file_name):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output

def load_poly_data(file_name):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()
    return output

def unstructured_grid_data_to_poly_data(unstructured_grid_data):
    filter = vtk.vtkDataSetSurfaceFilter()
    filter.SetInputData(unstructured_grid_data)
    filter.Update()
    poly_data = filter.GetOutput()
    return poly_data, filter

def get_sdf(target, boundary):
    nbrs = NearestNeighbors(n_neighbors=1).fit(boundary)
    dists, indices = nbrs.kneighbors(target)
    neis = np.array([boundary[i[0]] for i in indices])
    dirs = (target - neis) / (dists + 1e-8)
    return dists.reshape(-1), dirs

def get_normal(unstructured_grid_data):
    poly_data, surface_filter = unstructured_grid_data_to_poly_data(unstructured_grid_data)
    normal_filter = vtk.vtkPolyDataNormals()
    normal_filter.SetInputData(poly_data)
    normal_filter.SetAutoOrientNormals(1)
    normal_filter.SetConsistency(1)
    normal_filter.SetComputeCellNormals(1)
    normal_filter.SetComputePointNormals(0)
    normal_filter.Update()

    unstructured_grid_data.GetCellData().SetNormals(normal_filter.GetOutput().GetCellData().GetNormals())
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(unstructured_grid_data)
    c2p.Update()
    unstructured_grid_data = c2p.GetOutput()
    normal = vtk_to_numpy(c2p.GetOutput().GetPointData().GetNormals()).astype(np.double)
    normal /= (np.max(np.abs(normal), axis=1, keepdims=True) + 1e-8)
    normal /= (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    if np.isnan(normal).sum() > 0:
        print(np.isnan(normal).sum())
        print("recalculate")
        return get_normal(unstructured_grid_data)  
    return normal


def get_edges(unstructured_grid_data, points):
    edge_indices = set()
    cell_array = unstructured_grid_data.GetPolys() if isinstance(unstructured_grid_data, vtk.vtkPolyData) \
                 else unstructured_grid_data.GetCells()

    id_list = vtk.vtkIdList()
    for i in range(unstructured_grid_data.GetNumberOfCells()):
        unstructured_grid_data.GetCellPoints(i, id_list)
        ids = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
        for u, v in itertools.permutations(ids, 2):
            edge_indices.add((u, v))

    edges = [[], []]
    for u, v in edge_indices:
        edges[0].append(tuple(points[u]))
        edges[1].append(tuple(points[v]))
    return edges

def get_edge_index(pos, edges_press, edges_velo):
    indices = {tuple(pos[i]): i for i in range(len(pos))}
    edges = set()
    for i in range(len(edges_press[0])):
        edges.add((indices[edges_press[0][i]], indices[edges_press[1][i]]))
    for i in range(len(edges_velo[0])):
        edges.add((indices[edges_velo[0][i]], indices[edges_velo[1][i]]))
    edge_index = np.array(list(edges)).T
    return edge_index

def get_induced_graph(data, idx, num_hops):
    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=idx, num_hops=num_hops, edge_index=data.edge_index,
                                                  relabel_nodes=True)
    return Data(x=data.x[subset], y=data.y[idx], edge_index=sub_edge_index)

def pc_normalize(pc):
    centroid = torch.mean(pc, axis=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def get_shape(data, max_n_point=8192, normalize=True, use_height=False):
    surf_indices = torch.where(data.surf)[0].tolist()

    if len(surf_indices) > max_n_point:
        surf_indices = np.array(random.sample(range(len(surf_indices)), max_n_point))

    shape_pc = data.pos[surf_indices].clone()

    if normalize:
        shape_pc = pc_normalize(shape_pc)

    if use_height:
        gravity_dim = 1
        height_array = shape_pc[:, gravity_dim:gravity_dim + 1] - shape_pc[:, gravity_dim:gravity_dim + 1].min()
        shape_pc = torch.cat((shape_pc, height_array), axis=1)

    return shape_pc

def create_edge_index_radius(data, r, max_neighbors=32):
    data.edge_index = nng.radius_graph(x=data.pos, r=r, loop=True, max_num_neighbors=max_neighbors)
    return data

def get_sdf_from_mesh(target_pts, surface_mesh):
    sdf, _, _, normals = igl.signed_distance(
        target_pts,
        surface_mesh.vertices,
        surface_mesh.faces.astype(np.int32),
        return_normals=True
    )
    return sdf, normals


class GraphDataset(Dataset):
    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None, coef_norm=None, valid_list=None):
        super().__init__()
        self.datalist = datalist
        self.use_height = use_height
        self.coef_norm = coef_norm
        self.valid_list = valid_list
        if not use_cfd_mesh:
            assert r is not None
            for i in range(len(self.datalist)):
                self.datalist[i] = create_edge_index_radius(self.datalist[i], r)

    def len(self):
        return len(self.datalist)

    def get(self, idx):
        data = self.datalist[idx]
        shape = get_shape(data, use_height=self.use_height)
        if self.valid_list is None:
            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \
                data.edge_index
        else:
            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \
                data.edge_index, self.valid_list[idx]


def get_edge_index_from_mesh(unstructured_grid_data, mesh_points, unified_pos):

    mesh_arr = np.ascontiguousarray(mesh_points, dtype=np.float32)
    pos_arr = np.ascontiguousarray(unified_pos, dtype=np.float32)

    def hashable(arr):
        return arr.view([('', arr.dtype)] * arr.shape[1]).ravel()

    mesh_hash = hashable(mesh_arr)
    pos_hash = hashable(pos_arr)

    sort_idx = np.argsort(pos_hash)
    inv_map = np.searchsorted(pos_hash[sort_idx], mesh_hash)
    mapped_indices = sort_idx[inv_map]

    valid = pos_hash[sort_idx][inv_map] == mesh_hash
    mapped_indices[~valid] = -1

    edge_src = []
    edge_dst = []
    id_list = vtk.vtkIdList()

    for i in range(unstructured_grid_data.GetNumberOfCells()):
        unstructured_grid_data.GetCellPoints(i, id_list)
        ids = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
        local = mapped_indices[ids]

        if np.any(local < 0):
            continue

        edge_src.extend(np.repeat(local, len(local)))
        edge_dst.extend(np.tile(local, len(local)))

    edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
    return edge_index

def unsigned_distance_batched(mesh, points, batch_size=1_000_000):
    from trimesh.proximity import ProximityQuery
    pq = ProximityQuery(mesh)
    dists = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        d = pq.on_surface(batch)[1]
        dists.append(d)
    return np.concatenate(dists).astype(np.float32)

def compute_sdf_from_mesh(target_pts, surf_mesh, method: str = "igl"):
    method = (method or "mesh_signed").lower()
    if method == "mesh_signed":
        sdf, _, _, normals = igl.signed_distance(
            target_pts.astype(np.float64),
            surf_mesh.vertices.astype(np.float64),
            surf_mesh.faces.astype(np.int32),
            return_normals=True
        )
        return sdf.astype(np.float32), normals.astype(np.float32)

    elif method == "pc_unsigned":
        sdf_unsigned = unsigned_distance_batched(surf_mesh, target_pts)
        _, _, _, normals = igl.signed_distance(
            target_pts.astype(np.float64),
            surf_mesh.vertices.astype(np.float64),
            surf_mesh.faces.astype(np.int32),
            return_normals=True
        )
        return sdf_unsigned.astype(np.float32), normals.astype(np.float32)

    else:
        raise ValueError(f"Unknown SDF method: {method}")