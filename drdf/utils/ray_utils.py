import pdb

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from numpy.core.fromnumeric import clip
from trimesh.ray.ray_pyembree import RayMeshIntersector

from ..utils import geometry_utils
from ..utils import plt_vis as plt_vis_utils


def signed_zero_clip(data, a_max=None, a_min=None):
    data_sign = np.sign(data)
    data_sign[data_sign == 0] = 1.0
    data_clip = data_sign * np.clip(np.abs(data), a_max=a_max, a_min=a_min)
    return data_clip


# def get_ray_dir(points, RT, Kndc=np.eye(4)):
#     # points_tfs = geometry_utils.transform_points(points.transpose(), RT)
#     # depth = np.clip(np.abs(points_tfs[2:3, :]), a_max=10, a_min=1E-4)
#     # depth_sign = np.sign(points_tfs[2:3, :])
#     # depth = depth * depth_sign

#     # img_pts = points_tfs / depth
#     invRT = np.linalg.inv(RT)
#     # invKndc = np.linalg.inv(Kndc)
#     # local_ray_dir, _ = self.convert_pts_to_rays2(points.transpose(), RT)

#     pts_cam = geometry_utils.transform_points(points.transpose(),
#                                               RT).transpose()
#     hitting_depth = pts_cam[:, 2]
#     hitting_depth_clip = signed_zero_clip(hitting_depth, a_min=1E-4, a_max=None)
#     local_ray_dir = pts_cam / hitting_depth_clip[:, None]
#     local_ray_dir = local_ray_dir / (
#         1E-5 + np.linalg.norm(local_ray_dir, axis=1)[:, None]
#     )
#     start_point = pts_cam - (hitting_depth[:, None]) * local_ray_dir
#     world_start_points = geometry_utils.transform_points(
#         start_point.transpose(), invRT
#     ).transpose()
#     ray_dir = 1 * (points - world_start_points)
#     ray_dir = ray_dir / (1E-5 + np.linalg.norm(ray_dir, axis=1)[:, None])
#     return ray_dir
#     # img_pts, hitting_depth = self.convert_pts_to_rays(points.transpose(), RT, Kndc)
#     # img_pts = (1.0) * img_pts
#     # pdb.set_trace()
#     # img_pts_world = self.convert_rays_to_world(img_pts, invRT, invKndc)
#     # ray_dir = points.transpose() - img_pts_world
#     # ray_dir = ray_dir / (1E-5 + np.linalg.norm(ray_dir, axis=1)[:, None])
#     # return ray_dir.transpose(1, 0)
"""
Basic function to compute distance along the ray given a mesh, points, ray_dir
"""


def basic_distance_along_ray(
    mesh: trimesh.Trimesh,
    points: np.array,
    ray_dirs: np.array,
    mesh_intersector: RayMeshIntersector = None,
    embree: bool = True,
    clip_distance: float = None,
):

    if mesh_intersector is None:
        mesh_intersector = RayMeshIntersector(mesh)

    def compute_distance_along_ray(points, rays, clip_distance=None):
        locations, index_ray, index_tri = mesh_intersector.intersects_location(
            points, rays, multiple_hits=False
        )
        intersect_locations = points * 0
        intersect_locations[index_ray] = locations
        valid_intersects = points[:, 0] * 0
        valid_intersects[index_ray] = 1
        tri_ids = (points[:, 0] * 0 - 1).astype(int)
        tri_ids[index_ray] = index_tri
        distance = np.linalg.norm(points - intersect_locations, axis=1)
        distance = distance * valid_intersects + (1 - valid_intersects) * 0

        if clip_distance is not None:
            assert type(clip_distance) == float, "clip distance should be of type float"
            invalid_int = (1 - valid_intersects).astype(bool)
            intersect_locations[invalid_int, :] = (
                points[invalid_int, :] + rays[invalid_int, :] * clip_distance
            )
            valid_intersects[invalid_int] += 1.0
            distance[invalid_int] = clip_distance

        return distance, intersect_locations, valid_intersects, tri_ids

    dist, int_loc, valid_int, tri_ids = compute_distance_along_ray(
        points, ray_dirs, clip_distance=clip_distance
    )
    return dist, int_loc, valid_int, tri_ids


def get_special_ray_distances(
    mesh: trimesh.Trimesh,
    points: np.array,
    rays: np.array,
    unsigned_ray_dist: bool = False,
    signed: bool = False,
    mesh_intersector=None,
    **kwargs,
):

    if mesh_intersector is None:
        assert mesh is not None, "mesh input is necessary"
        mesh_intersector = RayMeshIntersector(mesh)

    dist, int_loc, valid_int, tri_ids = basic_distance_along_ray(
        mesh, points, rays, mesh_intersector, **kwargs
    )
    if unsigned_ray_dist:
        dist_bid, int_loc_bid, valid_int_bid, tri_ids_bid = basic_distance_along_ray(
            mesh, points, -1 * rays, mesh_intersector, **kwargs
        )
        min_selector = (dist + (1 - valid_int) * 10) > (
            dist_bid + (1 - valid_int_bid) * 10
        )
        min_selector = min_selector.astype(np.float32)
        int_loc = (
            int_loc * (1 - min_selector[:, None]) + min_selector[:, None] * int_loc_bid
        )
        valid_int = np.logical_or(valid_int, valid_int_bid)
        tri_ids = tri_ids * (1 - min_selector) + tri_ids * min_selector
        dist = dist * (1 - min_selector) + min_selector * dist_bid
    elif signed:
        # DRDF
        dist_bid, int_loc_bid, valid_int_bid, tri_ids_bid = basic_distance_along_ray(
            mesh, points, -1 * rays, mesh_intersector, **kwargs
        )
        min_selector = (dist + (1 - valid_int) * 10) > (
            dist_bid + (1 - valid_int_bid) * 10
        )
        min_selector = min_selector.astype(np.float32)
        int_loc = (
            int_loc * (1 - min_selector[:, None]) + min_selector[:, None] * int_loc_bid
        )
        valid_int = np.logical_or(valid_int, valid_int_bid)
        tri_ids = tri_ids * (1 - min_selector) + tri_ids * min_selector
        dist = dist * (1 - min_selector) + min_selector * (-1) * dist_bid

    return dist, int_loc, valid_int, tri_ids


def get_camera_ray_dir(points: np.array, RT: np.array, return_hitting_depth=False):
    """[summary]

    Args:
        points (np.array): N x 3 matrix
        RT (np.array): 4 x 4  matrix

    Returns:
        [type]: [description]
    """

    invRT = np.linalg.inv(RT)
    pts_cam = geometry_utils.transform_points(points.transpose(), RT).transpose()
    hitting_depth = pts_cam[:, 2]
    hitting_depth_clip = signed_zero_clip(hitting_depth, a_min=1e-4, a_max=None)
    local_ray_dir = pts_cam / hitting_depth_clip[:, None]
    local_ray_dir = local_ray_dir / (
        1e-5 + np.linalg.norm(local_ray_dir, axis=1)[:, None]
    )

    start_point = pts_cam - (1 + hitting_depth[:, None]) * local_ray_dir
    world_start_points = geometry_utils.transform_points(
        start_point.transpose(), invRT
    ).transpose()
    ray_dir = 1 * (points - world_start_points)
    ray_dir = ray_dir / (1e-8 + np.linalg.norm(ray_dir, axis=1)[:, None])
    # ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=1)[:, None]
    if return_hitting_depth:
        return ray_dir, hitting_depth
    else:
        return ray_dir


def get_axis_ray_dir(points, axis="X"):
    ray_dir = points * 0

    if axis == "X":
        ray_dir[..., 0] = 1
    elif axis == "Y":
        ray_dir[..., 1] = 1
    elif axis == "Z":
        ray_dir[..., 2] = 1
    else:
        assert False, "incorrect ray dir only supports X, Y , Z"

    return ray_dir


# def distance_along_ray(
#     mesh: trimesh.Trimesh,
#     points: np.array,
#     RT=None,
#     Kndc=None,
#     rays=None,
#     embree=True,
#     bidirectional=False,
#     signed=False,
#     mesh_intersector=None,
#     clip_distance=None,
# ):
#     if mesh_intersector is None:
#         if embree:
#             mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
#         else:
#             mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
#     else:
#         assert mesh is None, 'mesh and mesh_intersector given as input'

#     assert RT is not None or rays is not None, 'RT or rays should be not None'

#     if rays is None:
#         if type(points) == dict:
#             keys = points.keys()
#             rays = {key: None for key in keys}

#             for key in keys:
#                 rays[key] = get_ray_dir(points[key], RT, Kndc)
#                 points[key] = points[key] - rays[key] * 0.01
#         else:
#             rays = get_ray_dir(points, RT, Kndc)
#             points = points - rays * 0.01

#     def compute_distance_along_ray(points, rays, clip_distance=None):
#         locations, index_ray, index_tri = mesh_intersector.intersects_location(
#             points, rays, multiple_hits=False
#         )
#         intersect_locations = points * 0
#         intersect_locations[index_ray] = locations
#         valid_intersects = points[:, 0] * 0
#         valid_intersects[index_ray] = 1
#         tri_ids = (points[:, 0] * 0 - 1).astype(np.int)
#         tri_ids[index_ray] = index_tri
#         distance = np.linalg.norm(points - intersect_locations, axis=1)
#         distance = distance * valid_intersects + (1 - valid_intersects) * 0

#         if clip_distance is not None:
#             assert type(
#                 clip_distance
#             ) == float, 'clip distance should be of type float'
#             invalid_int = (1 - valid_intersects).astype(np.bool)
#             intersect_locations[invalid_int, :] = points[
#                 invalid_int, :] + rays[invalid_int, :] * clip_distance
#             valid_intersects[invalid_int] += 1.0
#             distance[invalid_int] = clip_distance

#         return distance, intersect_locations, valid_intersects, tri_ids

#     if type(points) == dict:
#         keys = points.keys()
#         dist = {key: None for key in keys}
#         int_loc = {key: None for key in keys}
#         valid_int = {key: None for key in keys}
#         tri_ids = {key: None for key in keys}

#         for key in points.keys():
#             dist[key], int_loc[key], valid_int[key], tri_ids[
#                 key] = compute_distance_along_ray(points[key], rays[key])

#             if bidirectional:
#                 dist_bid, int_loc_bid, valid_ind_bid, tri_ids_bid = compute_distance_along_ray(
#                     points[key], -1 * rays[key]
#                 )
#                 min_selector = dist[key] > dist_bid

#         return dist, int_loc, valid_int, tri_ids
#     else:
#         dist, int_loc, valid_int, tri_ids = compute_distance_along_ray(
#             points, rays, clip_distance=clip_distance
#         )

#         ## choose the minimum distance. Non oriented.
#         if bidirectional:
#             dist_bid, int_loc_bid, valid_int_bid, tri_ids_bid = compute_distance_along_ray(
#                 points, -1 * rays, clip_distance=clip_distance
#             )
#             min_selector = (dist + (1 - valid_int) *
#                             10) > (dist_bid + (1 - valid_int_bid) * 10)
#             min_selector = min_selector.astype(np.float32)
#             int_loc = int_loc * (1 - min_selector[:, None]
#                                  ) + min_selector[:, None] * int_loc_bid
#             valid_int = np.logical_or(valid_int, valid_int_bid)
#             tri_ids = tri_ids * (1 - min_selector) + tri_ids * min_selector
#             dist = dist * (1 - min_selector) + min_selector * dist_bid

#         if signed:
#             dist_bid, int_loc_bid, valid_int_bid, tri_ids_bid = compute_distance_along_ray(
#                 points, -1 * rays, clip_distance=clip_distance
#             )
#             min_selector = (dist + (1 - valid_int) *
#                             10) > (dist_bid + (1 - valid_int_bid) * 10)
#             min_selector = min_selector.astype(np.float32)
#             int_loc = int_loc * (1 - min_selector[:, None]
#                                  ) + min_selector[:, None] * int_loc_bid
#             valid_int = np.logical_or(valid_int, valid_int_bid)
#             tri_ids = tri_ids * (1 - min_selector) + tri_ids * min_selector
#             dist = dist * (1 - min_selector) + min_selector * (-1) * dist_bid

#         return dist, int_loc, valid_int, tri_ids
"""
points : 3 x N
RT : 4 x 4
"""


def convert_pts_to_rays2(points, RT):
    pts_cam = geometry_utils.transform_points(points, RT)
    hitting_depth = pts_cam[2, :]
    hitting_depth_clip = signed_zero_clip(hitting_depth, a_min=1e-4, a_max=None)
    local_ray_dir = pts_cam / hitting_depth_clip[None, :]
    return local_ray_dir, hitting_depth


def convert_rays_to_world2(pts, invRT):
    pts = geometry_utils.transform_points(pts, invRT)
    return pts


def compute_all_interesections_point(point, mesh_intersector, RT):
    ## point ## (3)
    ## Works on  single 3D point.
    ## this function needs to work recursively to find all intersections.\
    int_locs = []
    int_dists = []
    point = point[None]
    RT = RT.numpy()
    while True:
        ray_dir = get_camera_ray_dir(
            point,
            RT,
        )
        temp_point = point + 0.05 * ray_dir

        dist, int_loc, valid_int, tri_ids = get_special_ray_distances(
            mesh=None,
            points=temp_point,
            rays=ray_dir,
            mesh_intersector=mesh_intersector,
        )
        point = int_loc
        if (len(valid_int) > 0) and valid_int[0] > 0:
            int_locs.append(int_loc)
            int_dists.append(dist)
        else:
            break
    if len(int_dists) > 0:
        int_dists = np.cumsum(np.concatenate(int_dists))
    else:
        int_dists = np.array(int_dists)

    return int_dists


def compute_all_intersection(points, mesh, RT, max_ints=5):
    mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
        trimesh.Trimesh(vertices=mesh["verts"], faces=mesh["faces"])
    )
    all_int_dists = []
    for px in range(points.shape[1]):
        zero_ints = np.zeros(max_ints)
        point = points[:, px]
        int_dists = compute_all_interesections_point(point, mesh_intersector, RT)
        nints = min(max_ints, len(int_dists))
        zero_ints[0 : min(max_ints, len(int_dists))] = int_dists[0:nints]
        all_int_dists.append(zero_ints)
    all_int_dists = np.stack(all_int_dists)
    return all_int_dists


def compute_view_frustrum(RT: np.array, Kndc: np.array, z_max):
    """[summary]

    Args:
        points (np.array): [N x 3 array of points]
        rays (np.array): [N x 3 array of ray dirs] --> world to cam
        RT (np.array): [4 x 4 ext matrix]
        Kndc (np.array): [3 x 3 int matrix]
    """
    ## complete this function
    im_h = 2
    im_w = 2

    view_frust_pts = np.array(
        [
            (np.array([0, -1, -1, 1, 1]) - Kndc[0, 2])
            * np.array([0, z_max, z_max, z_max, z_max])
            / Kndc[0, 0],
            (np.array([0, -1, 1, -1, 1]) - Kndc[1, 2])
            * np.array([0, z_max, z_max, z_max, z_max])
            / Kndc[1, 1],
            np.array([0, z_max, z_max, z_max, z_max]),
        ]
    )

    view_frust_pts = geometry_utils.transform_points(view_frust_pts, np.linalg.inv(RT))
    # temp = geometry_utils.transform_points(view_frust_pts, RT)
    # temp_img = geometry_utils.perspective_transform(temp, Kndc)
    # view_frust_pts = np.matmul(cam_pose, view_frust_pts)
    return view_frust_pts

    # pdb.set_trace()
    # points_cam = geometry_utils.transform_points(points.transpose(), RT)
    # points_img = geometry_utils.perspective_transform(points_cam, Kndc)
    # points_img = points_img.transpose()

    # rays_cam = geometry_utils.transform_points(rays.transpose(), RT)
    # rays_img = geometry_utils.perspective_transform(rays_cam, Kndc)
    # rays_img = rays_img.transpose()

    # pdb.set_trace()
    return


def signed_divide(tensor1, tensor2):
    tensor2_sign = np.sign(tensor2)
    eps = 1e-5
    tensor2_sign[tensor2 == 0] = 1
    tensor2 = tensor2_sign * np.clip(np.abs(tensor2), a_min=eps, a_max=None)
    return tensor1 / tensor2


def plot_distance_func(z, dist_func, marker=".", color="r", **kwargs_plot):
    plt.scatter(z, dist_func, marker=marker, color=color)
    img = plt_vis_utils.plt_formal_to_image(**kwargs_plot)
    return img
