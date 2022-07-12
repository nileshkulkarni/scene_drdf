import copy
import pdb
import typing
from typing import Dict

import numpy as np
import trimesh


def convert_points_to_mesh(points, color=[255, 0, 0], radius=0.01):

    icosahedron = trimesh.creation.icosahedron()
    icosahedron.vertices *= radius
    mesh = trimesh.Trimesh()

    for point in points:
        point_mesh = copy.deepcopy(icosahedron)
        point_mesh.vertices += point[None, :]
        mesh = mesh + point_mesh

    color = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=color)
    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, visual=color)
    return mesh


def color_ray_with_signature(points, point_color, radius=0.01):

    icosahedron = trimesh.creation.icosahedron()
    icosahedron.vertices *= radius
    mesh = trimesh.Trimesh()
    vertex_colors_all = []
    for px, point in enumerate(points):
        point_mesh = copy.deepcopy(icosahedron)
        point_mesh.vertices += point[None, :]
        visuals = trimesh.visual.color.ColorVisuals(
            point_mesh, vertex_colors=point_color[px]
        )
        vertex_colors = visuals.vertex_colors
        point_mesh = trimesh.Trimesh(
            np.array(point_mesh.vertices) * 1,
            np.array(point_mesh.faces) * 1,
            vertex_color=vertex_colors,
        )
        vertex_colors_all.append(np.array(vertex_colors))
        # trimesh.exchange.export.export_mesh(point_mesh, 'test.ply')
        # point_mesh.visual.vertex_colors = vertex_colors
        mesh = mesh + point_mesh
    vertex_colors_all = np.concatenate(vertex_colors_all)
    visuals = trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vertex_colors_all)
    mesh = trimesh.Trimesh(
        np.array(mesh.vertices) * 1,
        np.array(mesh.faces),
        vertex_colors=visuals.vertex_colors,
    )

    # trimesh.exchange.export.export_mesh(mesh, 'test.ply')
    return mesh


def save_mesh(mesh, file_name):
    trimesh.exchange.export.export_mesh(mesh, file_name)
    return


def save_point_cloud(points, file_name):
    mesh = trimesh.Trimesh(points)
    trimesh.exchange.export.export_mesh(mesh, file_name)
    return


def get_closest_pts(mesh, points):
    closest, distance, _ = trimesh.proximity.closest_point(mesh, points)
    return distance, closest


def visualize_ray_sig(ray_sig: Dict, camera_pose: Dict):
    cone = trimesh.creation.cone(radius=0.1, height=0.2)
    neg_z = np.eye(4)
    neg_z[2, 2] = -1
    cone.apply_transform(neg_z)
    RT = np.linalg.inv(np.array(camera_pose["RT"]))
    cone = cone.apply_transform(RT)
    # pdb.set_trace()
    return cone


def save_point_cloud(points, file_name):
    mesh = trimesh.Trimesh(points)
    trimesh.exchange.export.export_mesh(mesh, file_name)
    return


def save_mesh(mesh, file_name):
    trimesh.exchange.export.export_mesh(mesh, file_name)
    return
