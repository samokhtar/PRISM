# viz_utils.py
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

import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm  
from matplotlib.colors import Normalize
import plotly.graph_objects as go
import trimesh
import pandas as pd



def visualize_velocity_field(pos, sp, bbox = [[-512, 512], [-512, 512], [0, 1024]], slice_values = [0,0,1.25], epsilons = [5,5,1], grid_res = 512):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axes_list = ['x', 'y', 'z']
    labels = ['x', 'y', 'z']

    for i, axis in enumerate(axes_list):
        grid_values, x_min, x_max, y_min, y_max, remaining_axes = get_slice_grid(
            pos, sp, axis, slice_values[i], epsilons[i], bbox, grid_res
        )

        if axis in ['x', 'y']:
            grid_values = np.rot90(grid_values)

        im = axs[i].imshow(
            grid_values.T,
            origin='lower',
            extent=(x_min, x_max, y_min, y_max),
            cmap='viridis',
            aspect='equal'
        )

        xlabel = labels[remaining_axes[0]]
        ylabel = labels[remaining_axes[1]]
        axs[i].set_title(f'{axis.upper()} Slice @ {slice_values[i]}')
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def visualize_velocity_in_domain(x, y, n_pts=5000):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    rand_int = np.random.randint(0, x.shape[0],n_pts)

    sp = np.linalg.norm(y, axis=1)
    magnitude = sp[rand_int]
    norm = Normalize(vmin=np.min(magnitude), vmax=np.max(magnitude))
    colors = cm.viridis(norm(magnitude))  

    ax.quiver(x[rand_int,0], x[rand_int,1], x[rand_int,2], y[rand_int,0], y[rand_int,1], y[rand_int,2], colors=colors, length=10, normalize=True)
    ax.set_axis_off()
    plt.show()
    
def plot_srf_pressure(pos, surf, pres, views = [(20, 0),(20, 90),(20, 180),(20, 270)], vmin=-1, vmax=1):
    fig = plt.figure(figsize=(28, 7), constrained_layout=True)  
    axes = []
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        sc = ax.scatter(
            pos[:,0][surf==1], pos[:,1][surf==1], pos[:,2][surf==1],
            c=pres[surf==1], s=10, vmin=vmin, vmax=vmax, cmap='viridis'
        )
        ax.set_axis_off()
        elev, azim = views[i]
        ax.view_init(elev=elev, azim=azim)
        set_axes_equal(ax)
        axes.append(ax)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.7, aspect=20, location='right')
    cbar.set_label("Pressure")
    plt.show()

def set_axes_equal(ax):

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max([x_range, y_range, z_range])

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim3d([mid_x - max_range/2, mid_x + max_range/2])
    ax.set_ylim3d([mid_y - max_range/2, mid_y + max_range/2])
    ax.set_zlim3d([mid_z - max_range/2, mid_z + max_range/2])

def get_slice_grid(pos, values, axis, slice_value, epsilon, bbox, grid_res):
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    idx = axis_dict[axis]

    slice_mask = np.abs(pos[:, idx] - slice_value) <= epsilon

    bbox_mask = (
        (pos[:, 0] >= bbox[0][0]) & (pos[:, 0] <= bbox[0][1]) &
        (pos[:, 1] >= bbox[1][0]) & (pos[:, 1] <= bbox[1][1]) &
        (pos[:, 2] >= bbox[2][0]) & (pos[:, 2] <= bbox[2][1])
    )

    combined_mask = slice_mask & bbox_mask
    pos_slice = pos[combined_mask]
    values_slice = values[combined_mask]

    remaining_axes = [i for i in range(3) if i != idx]
    xy = pos_slice[:, remaining_axes]

    x_min, y_min = xy.min(axis=0)
    x_max, y_max = xy.max(axis=0)
    xg = np.linspace(x_min, x_max, grid_res)
    yg = np.linspace(y_min, y_max, grid_res)
    grid_x, grid_y = np.meshgrid(xg, yg)

    grid_values = griddata(xy, values_slice, (grid_x, grid_y), method='linear')
    grid_values = np.nan_to_num(grid_values, nan=0.0)

    return grid_values, x_min, x_max, y_min, y_max, remaining_axes

def plot_points_mesh(mesh, points, values, bbox, cmin_v=None, cmax_v=None, flatshading: bool = False):

    cmin = np.min(values) if cmin_v is None else cmin_v
    cmax = np.max(values) if cmax_v is None else cmax_v

    x, y, z = mesh.vertices.T
    i, j, k = mesh.faces.T

    mesh_plot = go.Mesh3d(
        x=x,
        y=z,
        z=y,
        i=i,
        j=j,
        k=k,
        color='grey',
        opacity=0.2,
    )

    scatter_plot = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 2],
        z=points[:, 1],
        mode='markers',
        marker=dict(
            color=values,
            colorscale='Plasma',
            size=2,
            opacity=0.8,
            cmin=cmin,
            cmax=cmax,
            showscale=True
        ),
    )

    fig = go.Figure([mesh_plot, scatter_plot])

    fig.update_traces(
        flatshading=flatshading,
        lighting=dict(specular=1.0),
        selector=dict(type="mesh3d")
    )

    fig.update_scenes(
        xaxis_title_text='X',
        yaxis_title_text='Y',
        zaxis_title_text='Z',
        xaxis_showbackground=False,
        yaxis_showbackground=False,
        zaxis_showbackground=False,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True, range=bbox[0]),
            yaxis=dict(visible=True, range=bbox[2]),
            zaxis=dict(visible=True, range=bbox[1]),
            aspectmode='cube',  
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
        ),
        margin=dict(r=5, l=5, b=5, t=5),
        height=700,
        showlegend=True
    )

    return fig

def visualize_mesh(vertices: np.ndarray, faces: np.ndarray, flatshading: bool = False, height=800, width=800, color="pink"):


    x, y, z = vertices.T
    i, j, k = faces.T

    range_x = x.max() - x.min()
    range_y = y.max() - y.min()
    range_z = z.max() - z.min()

    max_range = max(range_x, range_y, range_z)
    aspect_ratio = dict(
        x=range_x / max_range,
        y=range_z / max_range,  
        z=range_y / max_range,
    )

    fig = go.Figure([
        go.Mesh3d(
            x=x,
            y=z,  
            z=y,
            i=i,
            j=j,
            k=k,
            color=color
        )
    ])

    fig.update_traces(
        flatshading=flatshading,
        lighting=dict(specular=1.0),
        selector=dict(type="mesh3d")
    )

    fig.update_layout(
        margin=dict(r=5, l=5, b=5, t=5),
        scene=dict(
            aspectmode="manual",        
            aspectratio=aspect_ratio,    
            xaxis_showspikes=False,
            yaxis_showspikes=False,
            zaxis_showspikes=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        scene_camera=dict(eye=dict(x=0.5, y=4, z=1.5)),
        width=width,
        height=height
    )

    return fig

def normalize_mesh(mesh, bbox):
    norm_vertices = (mesh.vertices-bbox[:,0])/(bbox[:,1]-bbox[:,0])
    norm_mesh = trimesh.Trimesh(vertices=norm_vertices, faces=mesh.faces)
    return norm_mesh