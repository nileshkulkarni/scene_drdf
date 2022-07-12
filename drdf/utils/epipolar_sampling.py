import pdb
from enum import Enum
from inspect import signature

import numpy as np
import torch
from sympy import intersection

from . import geometry_utils
from . import image as image_utils
from . import ray_utils, sal_utils, tensor_utils


class RayClass(Enum):
    VISIBLE = 1
    OCCLUDED = 2
    UNCERTRAIN = 4
    DISCONTUNITY = 8


def sample_occluded_rays(
    ref_img_data, nbr_img_data, num_rays=32, use_cuda=False, jitter_grid=True
):

    ref_RT = ref_img_data["RT"]
    ref_kNDC = ref_img_data["kNDC"]
    ref_depth = ref_img_data["depth"]

    aux_RT = nbr_img_data["RT"]
    aux_kNDC = nbr_img_data["kNDC"]
    aux_depth = nbr_img_data["depth"]

    points_aux, valid_points_aux, ndc_pts = geometry_utils.convert_depth_pcl(
        aux_depth,
        aux_RT,
        aux_kNDC,
        use_cuda=use_cuda,
        return_xyz=True,
        hw_scalar=0.5,
        jitter_grid=jitter_grid,
    )  ## points in world frame. -- N x 3 , N
    ndc_pts = ndc_pts.reshape(-1, 3)
    ndc_pts = tensor_utils.tensor_to_cuda(ndc_pts, cuda=use_cuda)

    points_ref = geometry_utils.covert_world_points_to_pixel(
        points_aux.transpose(1, 0), ref_RT, ref_kNDC, use_cuda=use_cuda
    )  ## points in image frame.
    points_ref = points_ref.transpose(1, 0)  ## (N, 3)
    valid_ref = (
        (torch.abs(points_ref[:, 0]) <= 1.0)
        * (torch.abs(points_ref[:, 1]) <= 1.0)
        * (torch.abs(points_ref[:, 2]) > 0.1)
    )

    if True:
        ref_depth = tensor_utils.tensor_to_cuda(ref_depth, cuda=use_cuda)

        points_ref_depth = image_utils.interpolate_depth(
            ref_depth[None], points_ref[None, :, 0:2]
        )[0, 0]
    if False:
        ref_depth_numpy = (ref_depth * 1).astype(np.float32)
        ref_depth = tensor_utils.tensor_to_cuda(ref_depth, cuda=use_cuda)
        points_ref_numpy = points_ref.numpy().astype(np.float32)
        points_ref_depth = image_utils.interpolate_depth_numpy(
            ref_depth_numpy[None], points_ref_numpy[None, :, 0:2]
        )[0, 0]
        points_ref_depth = tensor_utils.tensor_to_cuda(points_ref_depth, cuda=use_cuda)

    valid_ref_depth = points_ref_depth > 0.1
    all_valid = valid_ref * valid_ref_depth * valid_points_aux

    occluded_pts = (points_ref_depth - points_ref[:, 2]) < -0.05
    valid_occluded = (occluded_pts * all_valid).type(torch.bool)

    # occluded_locations = ndc_pts[valid_occluded == True]
    occluded_locations = points_aux[valid_occluded == True]
    occluded_locations = tensor_utils.tensor_to_numpy(occluded_locations)
    pt_tuple = sample_rays(occluded_locations, ref_RT, num_rays=num_rays)
    if pt_tuple is not None:
        points, ray_dir, ninds = pt_tuple

        points = np.concatenate([points, ray_dir], axis=-1)
    else:
        points = None
    return points


def sample_visible_rays(ref_img_data, num_rays=32, use_cuda=False, jitter_grid=True):

    ref_RT = ref_img_data["RT"]
    ref_kNDC = ref_img_data["kNDC"]
    ref_depth = ref_img_data["depth"]
    points, validity, ndc_pts = geometry_utils.convert_depth_pcl(
        ref_depth,
        ref_RT,
        ref_kNDC,
        use_cuda=use_cuda,
        return_xyz=True,
        hw_scalar=0.5,
        jitter_grid=jitter_grid,
    )
    points, ray_dir, ninds = sample_rays(points, ref_RT, num_rays=num_rays)
    points = np.concatenate([points, ray_dir], axis=-1)
    return points


def sample_rays(points_world, RT, num_rays, max_depth=8.0):
    camera_loc = np.linalg.inv(RT)[0:3, 3]
    if len(points_world) > num_rays:
        ninds = np.random.choice(len(points_world), num_rays, replace=False)
        points_world = points_world[ninds]
        ray_dir, hitting_depth = ray_utils.get_camera_ray_dir(
            points_world, RT, return_hitting_depth=True
        )
        sampled_depths = np.linspace(0.1, 1, 512)[None, :, None]
        sampled_depths = sampled_depths * max_depth
        points = camera_loc[None, None, :] + sampled_depths * ray_dir[:, None, :]

        ray_dir = points * 0 + ray_dir[:, None, :]

        if False:
            temp = points[0] - points_world[0][None, :]
            temp = temp / np.linalg.norm(temp, axis=1)[:, None]
            pdb.set_trace()
        # points = points_world[:,
        #                       None, :] - 1 * sampled_depths * ray_dir[:,
        #                                                               None, :]
        return points, ray_dir, ninds
    else:
        return None


def create_ray_weight(ray_sig, window, direction=False, sign=1):
    # pdb.set_trace()
    intersect_locs, _ = sal_utils.zero_crossings(ray_sig, window=window)
    ray_weight = np.zeros(ray_sig.shape)
    linspace_wt = np.linspace(0, 1, window)
    linspace_wt_constant = linspace_wt * 0 + 1.0
    triangle_wt = np.concatenate(
        [linspace_wt[:-1], linspace_wt_constant, linspace_wt[::-1]]
    )

    if direction:
        # triangle_wt = np.concatenate(
        #     [linspace_wt_constant[:-1], linspace_wt_constant, linspace_wt[::-1]]
        # )
        triangle_wt = np.concatenate(
            [
                linspace_wt_constant[:-1],
                linspace_wt_constant,
                linspace_wt_constant[::-1],
            ]
        )
        if sign == -1:
            triangle_wt = triangle_wt[::-1]

    half_len = (len(triangle_wt)) // 2
    half_len = int(half_len)
    for loc in intersect_locs:
        min_ind = max(loc - half_len, 0)
        max_ind = min(loc + half_len + 1, ray_sig.shape[0])
        clip_tri_inds_min = min_ind - loc + half_len
        clip_tri_inds_max = max_ind - loc + 1 + half_len
        ray_weight[min_ind:max_ind] += triangle_wt[
            clip_tri_inds_min:clip_tri_inds_max
        ]  ## reverse since after the zero crossing
        # ray_weight[loc] += 1
    return ray_weight


def create_ray_signature_from_distances(points, RT, ray_distances, validity, window=5):
    ray_distances = ray_distances * 1
    ray_distances[validity == False] = np.nan
    intersect_locs, _ = sal_utils.zero_crossings(ray_distances, window=window)
    # intersect_locs = np.where(np.abs(ray_sig) < 0.005)[0]

    normal_weight, normal_sign = scale_with_normal_dir(
        points,
        RT,
    )
    sign = np.sign(np.sum(normal_sign))
    ray_class_sig = (ray_distances <= 0) * RayClass.VISIBLE.value
    ray_class_sig += (ray_distances > 0) * RayClass.OCCLUDED.value
    ray_class_sig += np.isnan(ray_distances) * RayClass.UNCERTRAIN.value
    ray_discont = detect_discontunity(ray_distances, window)

    ray_class_sig += ray_discont * RayClass.DISCONTUNITY.value

    ray_weight = create_ray_weight(
        ray_distances, window=window, direction=True, sign=sign
    )

    return intersect_locs, ray_class_sig, ray_weight, normal_weight, normal_sign


def detect_discontunity(ray_dist, window):
    ray_discont = np.abs(ray_dist[:-1] - ray_dist[1:]) > 0.4
    inds = np.where(ray_discont)[0]
    temp_ray_dist = ray_dist[:-1]
    for ind in inds:
        ray_dist_local = temp_ray_dist[ind - window : ind + window] > 0
        # if len(ray_dist_local) == 23:
        #     pdb.set_trace()
        ray_discont[ind - window : ind + window] = ray_dist_local

    final_ray_discont = (ray_dist * 0).astype(int)
    # pdb.set_trace()
    final_ray_discont[0:-1] = ray_discont
    return final_ray_discont


def scale_with_normal_dir(points, RT):
    invRT = np.linalg.inv(RT)
    cam_loc = invRT[:3, 3]
    dir_vec = points[-1] - points[0]
    dir_vec = dir_vec / (1e-5 + np.linalg.norm(dir_vec))
    z_vec = np.array([0, 0, 1]).reshape(3, -1)  ## camera looks in +z
    camera_pt = geometry_utils.transform_points(z_vec, invRT)[:, 0]
    cam_dir = camera_pt - cam_loc

    local_dirs = points - cam_loc[None, :]
    local_dirs = local_dirs / (1e-5 + np.linalg.norm(local_dirs, axis=1)[:, None])

    dot_p = np.sum(local_dirs * dir_vec[None, :], axis=1)
    ray_normal_wt = np.abs(dot_p)
    ray_dir_sign = np.sign(np.sum(cam_dir * dir_vec))
    normal_sign = np.sign(dot_p) * 0 + ray_dir_sign
    return ray_normal_wt, normal_sign


def compute_ray_signature(points, RT, Kndc, depth_img):
    """
    list of points along the ray
    """

    pts_ndc = geometry_utils.covert_world_points_to_pixel(points.transpose(), RT, Kndc)
    pts_ndc = pts_ndc.transpose(1, 0)

    pts_z = pts_ndc[:, 2]
    pts_depth = image_utils.interpolate_depth(
        depth_img[
            None,
        ],
        pts_ndc[:, None, 0:2],
    )[0, :, 0]

    pts_validity = image_utils.get_point_validity(pts_ndc[:, 0:2])
    depth_validity = image_utils.get_depth_validity(pts_depth)
    validity = np.logical_and(pts_validity, depth_validity)
    ray_dist = (
        pts_z - pts_depth
    )  ## +ve means ray behind the visible, -ve means ray in front of the visible,  0 means ray is intersecting!

    ray_dist = tensor_utils.tensor_to_numpy(ray_dist)
    validity = tensor_utils.tensor_to_numpy(validity)
    window = 16
    (
        inter_loc,
        ray_class_sig,
        ray_weight,
        normal_weight,
        normal_sign,
    ) = create_ray_signature_from_distances(points, RT, ray_dist, validity, window=16)
    sign = np.sign(np.sum(normal_sign))
    ray_dist = sign * ray_dist
    if False:
        import matplotlib.pyplot as plt

        cm = plt.cm.plasma
        depth_colored = cm(depth_img / 5)
        depth_colored = (depth_colored * 255).astype(np.uint8)

        temp = image_utils.draw_points_on_img(
            depth_colored[:, :, 0:3], tensor_utils.tensor_to_numpy(pts_ndc[:, 0:2])
        )

    # normal_weight, normal_sign = scale_with_normal_dir(
    #     points,
    #     RT,
    # )

    ray_dist_nan = ray_dist * 1
    ray_dist_nan[validity == False] = np.nan
    discont_inds = np.where(np.bitwise_and(ray_class_sig, RayClass.DISCONTUNITY.value))[
        0
    ]
    if len(discont_inds) > 0:
        ray_dist_nan[discont_inds] = np.nan  ## ignore discontinuity

    intersections = sal_utils.zero_crossings(
        ray_dist_nan, window=5, direction=True, alignment="neg2pos"
    )[0]

    ray_dist_nan = ray_dist_nan
    signature_data = {
        "dist": ray_dist,
        "dist_nan": ray_dist_nan,
        "validity": validity,
        "inter_loc": inter_loc,
        "ray_class_sig": ray_class_sig,
        "pts_depth": pts_depth,
        "pts_z": pts_z,
        "ray_weight": ray_weight,
        "normal_weight": normal_weight,
        "normal_sign": normal_sign,
        "points": points * 1,
        "intersections": intersections,
    }
    return signature_data


def compute_all_ray_signatures(sampled_rays, ref_imgdata, nbr_posed_img_data):
    num_nbrs = len(nbr_posed_img_data)

    rgbd_obs_lst = [ref_imgdata] + nbr_posed_img_data

    num_obs = len(rgbd_obs_lst)

    all_ray_signature_data = []
    aggregrated_ray_data = []
    for sx, sray in enumerate(sampled_rays):
        ray_signature_data_lst = []
        sray_points = sray[..., 0:3]
        for px, posed_data in enumerate(rgbd_obs_lst):

            RT = posed_data["RT"]
            Kndc = posed_data["kNDC"]
            depth_img = posed_data["depth_hr"]

            signature_data_px = compute_ray_signature(sray_points, RT, Kndc, depth_img)
            ray_signature_data_lst.append(signature_data_px)
        all_ray_signature_data.append(ray_signature_data_lst)

        sray_data = {}
        if True:
            ray_sigs = [pose_rays["dist_nan"] for pose_rays in ray_signature_data_lst]
            ray_sigs = np.stack(
                ray_sigs,
            )
            # ray_sigs[np.isnan(ray_sigs)] = 100.0

            all_intersections = np.concatenate(
                [pose_rays["intersections"] for pose_rays in ray_signature_data_lst]
            )
            ray_weights = np.stack(
                [pose_rays["ray_weight"] for pose_rays in ray_signature_data_lst]
            )
            normal_weights = np.stack(
                [pose_rays["normal_weight"] for pose_rays in ray_signature_data_lst]
            )
            normal_signs = np.stack(
                [pose_rays["normal_sign"] for pose_rays in ray_signature_data_lst]
            )

            ray_class_lst = [
                pose_rays["ray_class_sig"] for pose_rays in ray_signature_data_lst
            ]
            ray_class_lst = np.stack(ray_class_lst)
            # ray_weights = np.stack(ray_weights,)
            # all_weights = ray_weights * normal_weights
            all_weights = ray_weights * 1
            all_weights[np.isnan(ray_sigs)] = 0.0

            # weight = 1 ## this weight is function of normal direction to the ray -- implying camera views that are perpendicular to the ray add almost zero weight.
            invalid = np.sum(all_weights, axis=0) < 1e-10
            weight = all_weights / (
                1e-5
                + np.nansum(all_weights, axis=0)[
                    None,
                ]
            )

            ray_sigs_signed = ray_sigs * 1

            common_ray = ray_sigs * weight
            common_ray = np.nansum(common_ray, axis=0)
            common_ray[invalid] = np.nan

            common_ray_signed = ray_sigs_signed * weight
            common_ray_signed = np.nansum(common_ray_signed, axis=0)
            common_ray_signed[invalid] = np.nan

            intersections = sal_utils.zero_crossings(
                common_ray, window=16, direction=True, alignment="neg2pos"
            )[0]
            # intersections = sal_utils.zero_crossings(common_ray, window=5, )[0]

            ray_class_common = np.bitwise_or.reduce(ray_class_lst, axis=0).astype(int)

            if sx == 88:
                pdb.set_trace()
            intersections = merge_intersections(
                intersections=all_intersections.astype(int)
            )
            # print(intersections)

            sray_data["ray"] = sray
            sray_data["intersections"] = intersections
            sray_data["ray_class_common"] = ray_class_common
            aggregrated_ray_data.append(sray_data)
    return all_ray_signature_data, aggregrated_ray_data


def merge_intersections(intersections, window=16):
    intersections = np.unique(intersections, axis=0)
    intersections = np.sort(intersections)
    clusters = []
    current_cluster = None
    cluster_center = None
    for inter in intersections:
        if current_cluster is None:
            current_cluster = [inter]
            cluster_center = inter
        else:
            if np.abs(cluster_center - inter) < window:
                current_cluster.append(inter)
                cluster_center = int(np.mean(np.array(current_cluster), axis=0))
            else:
                cluster_center = int(np.mean(np.array(current_cluster), axis=0))
                clusters.append(cluster_center)
                current_cluster = [inter]
                cluster_center = inter
    if cluster_center is not None:
        clusters.append(cluster_center)
    return clusters


def sample_points_from_intersections_multiple_rays(
    ray_signatures_agg_lst,
    ref_imgdata,
):

    ref_RT = ref_imgdata["RT"]
    kNDC = ref_imgdata["kNDC"]

    sampled_points_lst = []
    for rx, ray_signature_agg in enumerate(ray_signatures_agg_lst):
        # if rx == 18:
        #     pdb.set_trace()
        sampled_points = sample_points_from_intersections(
            ray_signature_agg, ref_imgdata
        )

        if sampled_points is not None:
            sampled_points_lst.append(sampled_points)

    points, distances, ray_sig = collate_outputs(sampled_points_lst)
    return points, distances, ray_sig


def collate_outputs(tuple_lst):
    data = [k for k in zip(*tuple_lst)]
    data = (np.stack(k) for k in data)
    return data


def sample_points_from_intersections(ray_signature_agg, ref_imgdata):
    rays = ray_signature_agg["ray"]
    points = rays[..., 0:3]
    intersections_inds = ray_signature_agg["intersections"]
    RT = ref_imgdata["RT"]
    kNDC = ref_imgdata["kNDC"]
    ray_sig = ray_signature_agg["ray_class_common"]
    if len(intersections_inds) > 0:

        intersections_inds = np.array(intersections_inds)
        intersection_pts = points[intersections_inds]

        distances = compute_ray_distance(intersections_inds, intersection_pts, points)

        # ray_dir, hitting_depth = ray_utils.get_camera_ray_dir(
        #     intersection_pts, RT, return_hitting_depth=True
        # )

        # points = np.concatenate([intersection_pts, ray_dir], axis=1)
    else:
        return None
    return (rays, distances, ray_sig)


def compute_ray_distance(intersection_inds, intersections, points, signed=True):
    distances = np.linalg.norm(
        points[
            None,
        ]
        - intersections[:, None, :],
        axis=2,
    )
    min_inds = np.argmin(distances, axis=0)
    distances = points - intersections[min_inds]
    distances = np.linalg.norm(distances, axis=1)
    if signed:
        signs = np.sign(intersection_inds[min_inds] - np.arange(len(points)))
        signs[signs == 0] = 1
        distances = signs * distances
    return distances


def define_invisible_regions(
    distance_func,
    ray_sig,
    min_dist,
    max_dist,
):
    invisible_regions = ray_sig == RayClass.OCCLUDED
    invisible_regions_clipped = (distance_func > min_dist) * (distance_func <= 0)
    unknown_invisible_regions = np.logical_and(
        invisible_regions, np.logical_not(invisible_regions_clipped)
    )
    return unknown_invisible_regions


def define_missing_regions(
    distance_func,
    ray_sig,
):
    missing_regions = ray_sig == RayClass.UNCERTRAIN
    return missing_regions
