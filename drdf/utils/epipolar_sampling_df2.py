import pdb
from dis import disco
from enum import Enum
from turtle import distance
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np
from cv2 import merge
from loguru import logger
from rgbd_drdf.utils import depth_image

from . import depth_image as depth_image_utils
from . import geometry_utils
from . import image as image_utils
from . import sal_utils, tensor_utils


class RayEventType(Enum):
    INTERSECTION = 1
    OCCLUSION = 2
    DISOCCLUSION = 4
    START = 8
    END = 16


## assume a ray is going from left to right.
class RayFreeSpaceDirection(Enum):
    LEFT = 1
    RIGHT = 2


class RayEvent:
    def __init__(
        self, index: int, direction: RayFreeSpaceDirection, event_type: RayEventType
    ):
        self.index = index
        self.direction = direction
        self.event_type = event_type

    def __str__(self):
        dir_str = "-->" if self.direction == RayFreeSpaceDirection.RIGHT else "<--"
        print_str = f"({self.index}, {self.event_type}, {dir_str}) "
        return print_str

    def __eq__(self, other):
        return (
            self.index == other.index
            and self.direction == other.direction
            and self.event_type == other.event_type
        )


class RaySignature:
    def __init__(self, events: List[RayEvent] = []):
        self.evidence_count = 0
        if len(events) == 0:
            self.events = []
        else:
            self.events = [k for k in events]

    def add_event(self, ray_event: RayEvent):
        self.events.append(ray_event)

    def add_segment(self, segment):
        for event in segment.events:
            self.events.append(event)

    def sort_events(self):
        self.events.sort(
            key=lambda x: x.index + (x.direction == RayFreeSpaceDirection.RIGHT) * 0.5
        )

    def clean_events(
        self,
    ):
        self.sort_events()
        new_event_list = []

        if len(self.events) == 2:
            new_event_list = []
        else:
            past_event = None
            for ex, event in enumerate(self.events):
                if event.event_type == RayEventType.START:
                    if self.events[ex + 1].direction == RayFreeSpaceDirection.LEFT:
                        new_event = RayEvent(
                            index=event.index,
                            direction=event.direction,
                            event_type=RayEventType.OCCLUSION,
                        )
                        new_event_list.append(new_event)
                elif event.event_type == RayEventType.END:
                    if past_event.direction == RayFreeSpaceDirection.RIGHT:
                        new_event = RayEvent(
                            index=event.index,
                            direction=event.direction,
                            event_type=RayEventType.OCCLUSION,
                        )
                        new_event_list.append(new_event)
                else:
                    new_event = RayEvent(
                        index=event.index,
                        direction=event.direction,
                        event_type=event.event_type,
                    )
                    new_event_list.append(new_event)
                past_event = event
        self.events = new_event_list

    def __str__(
        self,
    ):
        self.sort_events()
        print_str = ""
        for event in self.events:
            print_str += str(event)
        return print_str

    def empty_ray(
        self,
    ):
        if len(self.events) == 2:
            if (self.events[0].event_type == RayEventType.START) and (
                self.events[1].event_type == RayEventType.END
            ):
                return True
            else:
                return False
        elif len(self.events) == 0:
            return True
        else:
            return False

    def __getitem__(self, index):
        return self.events[index]

    def __len__(self):
        return len(self.events)

    def __eq__(self, other):
        if len(self.events) != len(other.events):
            return False
        equal = True
        for i in range(len(self.events)):
            if self.events[i] != other.events[i]:
                equal = equal and False
        return equal


def get_normal_dir(points, RT):
    invRT = np.linalg.inv(RT)
    cam_loc = invRT[:3, 3]
    dir_vec = points[-1] - points[0]
    dir_vec = dir_vec / (1e-5 + np.linalg.norm(dir_vec))
    z_vec = np.array([0, 0, 1]).reshape(3, -1)  ## camera looks in +z
    camera_pt = geometry_utils.transform_points(z_vec, invRT)[:, 0]
    cam_dir = camera_pt - cam_loc
    normal_sign = np.sign(np.sum(cam_dir * dir_vec))
    return normal_sign


def get_frustum_clip_locations(pts_validity, ray_distances_visible):

    valid_inds_start = np.where((pts_validity * ray_distances_visible) == True)[0]
    valid_inds_end = np.where(pts_validity == True)[0]
    if len(valid_inds_start) == 0:
        start_ind = -1
    else:
        start_ind = valid_inds_start[0]

    if len(valid_inds_end) == 0:
        end_ind = len(pts_validity) * 2
    else:
        end_ind = valid_inds_end[-1]

    return start_ind, end_ind


def get_ray_events(points, RT, pts_z, pts_depths, pts_validity, depth_validity, window):
    """'
    :param points

    pts_validity: pts in the camera frustum
    depth_validity: pts with valid depth values
    """

    ## compute ray events
    ray_signature = RaySignature()
    ray_distances = pts_z - pts_depths
    ## ray_distances negative means points in front of the camera -- not occluded by anything.
    ray_distances_visible = ray_distances < 0
    frustum_clip_locations = get_frustum_clip_locations(
        pts_validity, ray_distances_visible
    )

    # frustum_clip_locations = (0, len(points) - 1)
    start_event = RayEvent(
        frustum_clip_locations[0], RayFreeSpaceDirection.RIGHT, RayEventType.START
    )
    end_event = RayEvent(
        frustum_clip_locations[1], RayFreeSpaceDirection.LEFT, RayEventType.OCCLUSION
    )
    # pdb.set_trace()
    # ray_signature.add_event(start_event)
    # ray_signature.add_event(end_event)

    ray_distances_non_nan = pts_z - pts_depths
    ray_distances = pts_z - pts_depths

    if False:
        normal_sign = get_normal_dir(
            points,
            RT,
        )
    # ray_distances = ray_distances * normal_sign
    # invalid_depths = np.where(1 - valid_depths(pts_depths))[
    #     0]  ## pts with zero depths

    validity = np.logical_and(pts_validity, depth_validity)
    ray_distances[np.logical_not(validity)] = np.nan
    ray_distances = fill_missing_nans(ray_distances, window=window)

    intersect_locs, _, signs = sal_utils.zero_crossings(
        ray_distances, window=1, return_sign=True
    )
    intersect_locs = np.array(intersect_locs)
    signs = np.array(signs)
    if len(intersect_locs) > 0:
        filter_mask = filter_locations(ray_distances, intersect_locs)
        intersect_locs = intersect_locs[filter_mask]
        signs = signs[filter_mask]
    discontunities = detect_discontunity(ray_distances, window=1)
    discontunities = np.array(discontunities)
    if len(discontunities) > 0:
        filter_mask = filter_locations(ray_distances, discontunities)
        discontunities = discontunities[filter_mask]
    # filtered_discontunities = []
    # for dist_ind in discontunities:
    #     if np.abs(ray_distances[dist_ind + 1]) < 1E-5 or np.isnan(
    #         ray_distances[dist_ind + 1] or np.abs(ray_distances[dist_ind + 1])
    #     ):
    #         continue
    #     else:
    #         filtered_discontunities.append(dist_ind)
    # discontunities = np.array(filtered_discontunities)
    ## this will give all possible events along the ray -- now classify them as occlusion or disocclusion or intersection
    # signs = signs * normal_sign
    first_event = None
    last_event = None
    for ix, intersect_loc in enumerate(intersect_locs):
        if len(discontunities) > 0:
            closest_discont = np.min(np.abs(discontunities - intersect_loc))
        else:
            closest_discont = np.inf
        if closest_discont < window:
            if signs[ix] > 0:
                free_space_direction = RayFreeSpaceDirection.LEFT
                event_type = RayEventType.OCCLUSION
            else:
                event_type = RayEventType.DISOCCLUSION
                free_space_direction = RayFreeSpaceDirection.RIGHT
            ray_event = RayEvent(intersect_loc, free_space_direction, event_type)
        else:
            ## this
            if signs[ix] > 0:
                free_space_direction = RayFreeSpaceDirection.LEFT
            else:
                free_space_direction = RayFreeSpaceDirection.RIGHT
            ray_event = RayEvent(
                intersect_loc, free_space_direction, RayEventType.INTERSECTION
            )

        if ray_event is not None:
            if first_event is None:
                first_event = ray_event
            ray_signature.add_event(ray_event)
            last_event = ray_event

    if len(ray_signature) > 0:
        if first_event.direction == RayFreeSpaceDirection.LEFT:
            if (
                np.nanmedian(
                    ray_distances[start_event.index : start_event.index + window // 2]
                )
                < 0
            ):
                start_event.event_type = RayEventType.DISOCCLUSION
            ray_signature.add_event(start_event)

        if last_event.direction == RayFreeSpaceDirection.RIGHT:
            median_vis = np.nanmedian(
                ray_distances[end_event.index - window // 2 : end_event.index]
            )
            if (
                (not (median_vis == np.nan))
                and (end_event.index < len(ray_distances))
                and median_vis < 0
            ):

                end_event.event_type = RayEventType.OCCLUSION
            else:
                # Try to find last event using the last visible part of ray after the open of the segment.
                if ray_signature[-1].direction == RayFreeSpaceDirection.RIGHT:
                    start_ind = ray_signature[-1].index
                    end_ind = np.where(ray_distances[start_ind:] < 0)[0]
                    if len(end_ind) > 0:
                        end_ind = end_ind[-1] + start_ind
                        end_event = RayEvent(
                            end_ind, RayFreeSpaceDirection.RIGHT, RayEventType.OCCLUSION
                        )
                    else:
                        pdb.set_trace()
                else:
                    pdb.set_trace()
            ray_signature.add_event(end_event)
    # ray_signature.clean_events()
    # pdb.set_trace()
    # if len(ray_signature) > 0:
    #     for rx in range(len(ray_signature)):
    #         if ray_signature[rx].event_type == RayEventType.START:
    #             pdb.set_trace()
    # if len(ray_signature) > 6:
    #     pdb.set_trace()

    return ray_signature


def fill_missing_nans(ray_dists, window=10):

    nan_inds = np.where(np.isnan(ray_dists))[0]
    components = []
    current_ind = None
    current_cluster = []
    cluster_lables = []
    cluster_ind = None
    # breakpoint()
    if len(nan_inds) > 0:
        differences = nan_inds[1:] - nan_inds[:-1]
        changes = np.where(differences > window)[0]

        components = []
        prev_change_ind = 0
        if len(changes) > 0:
            for change_ind in changes + 1:
                components.append(nan_inds[prev_change_ind:change_ind])
                prev_change_ind = change_ind
            components.append(nan_inds[prev_change_ind:])

        component_lens = np.array([len(component) for component in components])
        small_components = np.where(component_lens < window)[0]

        for cid in small_components:
            component = components[cid]
            for cx in component:
                ray_dists[cx] = 0

    return ray_dists


def fill_missing_nans_old(ray_dists, window=10):

    nan_inds = np.where(np.isnan(ray_dists))[0]
    components = []
    current_ind = None
    current_cluster = []
    cluster_lables = []
    cluster_ind = None
    breakpoint()

    for nan_ind in nan_inds:
        if current_ind is None:
            cluster_ind = 0
            current_ind = nan_ind
            current_cluster.append(nan_ind)
            cluster_lables.append(cluster_ind)
        else:
            if nan_ind - current_ind <= window:
                current_cluster.append(nan_ind)
                current_cluster.append(nan_ind)
                cluster_lables.append(cluster_ind)
            else:
                cluster_ind += 1
                components.append(current_cluster)
                current_cluster = []
                current_cluster.append(nan_ind)
                cluster_lables.append(cluster_ind)
            current_ind = np.mean(current_cluster[-window:])

    if len(current_cluster) > 0:
        components.append(current_cluster)

    for component in components:
        if len(component) < window:
            for cx in component:
                ray_dists[cx] = 0

    return ray_dists


def valid_depths(depths):
    return depths > 1e-3


def filter_locations(ray_dist, inds):
    filtered_inds = []
    for ind in inds:
        if (
            np.abs(ray_dist[ind + 1]) < 1e-5
            or np.isnan(ray_dist[ind + 1])
            or np.abs(ray_dist[ind]) < 1e-5
        ):
            filtered_inds.append(0)
            continue
        else:
            filtered_inds.append(1)
    return np.array(filtered_inds).astype(bool)


def detect_discontunity(ray_dist, window):
    length = len(ray_dist)
    ray_discont = np.abs(ray_dist[:-1] - ray_dist[1:]) > 0.1
    inds = np.where(ray_discont)[0]
    return inds


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

    pts_ndc = tensor_utils.tensor_to_numpy(pts_ndc)
    pts_validity = image_utils.get_point_validity(pts_ndc[:, 0:2])
    depth_validity = image_utils.get_depth_validity(pts_depth)
    validity = np.logical_and(pts_validity, depth_validity)
    ray_dist = (
        pts_z - pts_depth
    )  ## +ve means ray behind the visible, -ve means ray in front of the visible,  0 means ray is intersecting!

    pts_z = tensor_utils.tensor_to_numpy(pts_z)
    pts_depth = tensor_utils.tensor_to_numpy(pts_depth)
    ray_dist = tensor_utils.tensor_to_numpy(ray_dist)
    validity = tensor_utils.tensor_to_numpy(validity)
    window = 10
    timer = Timer()
    timer.tic()
    ray_events = get_ray_events(
        points, RT, pts_z, pts_depth, pts_validity, depth_validity, window=window
    )
    timer.toc()
    logger.debug(f"Time to compute ray events: {timer.get_time()}")
    # pdb.set_trace()
    # sign = np.sign(np.sum(normal_sign))
    # ray_dist = sign * ray_dist

    # ray_dist_nan = ray_dist * 1
    # ray_dist_nan[validity == False] = np.nan
    # discont_inds = np.where(
    #     np.bitwise_and(ray_class_sig, RayClass.DISCONTUNITY.value)
    # )[0]
    # if len(discont_inds) > 0:
    #     ray_dist_nan[discont_inds] = np.nan  ## ignore discontinuity

    # intersections = sal_utils.zero_crossings(
    #     ray_dist_nan, window=5, direction=True, alignment='neg2pos'
    # )[0]

    # ray_dist_nan = ray_dist_nan
    return ray_events


from ..utils.timer import Timer


def compute_all_ray_signatures(sampled_rays, ref_imgdata, nbr_posed_img_data):

    num_nbrs = len(nbr_posed_img_data)

    rgbd_obs_lst = [ref_imgdata] + nbr_posed_img_data

    num_obs = len(rgbd_obs_lst)

    all_ray_events = []
    aggregrated_ray_data = []
    all_ray_montaages = []
    timer = Timer()
    for sx, sray in enumerate(sampled_rays):
        ray_event_lst = []
        sray_points = sray[..., 0:3]
        timer.tic()
        for px, posed_data in enumerate(rgbd_obs_lst):
            RT = posed_data["RT"]
            Kndc = posed_data["kNDC"]
            depth_img = posed_data["depth_hr"]
            # if px == 19:
            #     breakpoint()
            timer.tic()
            ray_events = compute_ray_signature(sray_points, RT, Kndc, depth_img)
            timer.toc()
            t = timer.get_time()
            # logger.info('compute ray signature {}'.format(t))
            # print('posed ray sig {}'.format(px))
            # print(ray_events)
            ray_event_lst.append(ray_events)
        if False:
            montage_lst = visualize_raw_ray_events(
                sray_points, ray_event_lst, rgbd_obs_lst
            )
            montage = create_row_column_montange(montage_lst, (5, 6))
            imageio.imsave("test.png", montage)
            all_ray_montaages.append(montage)

        merged_ray_event = merge_all_ray_events(ray_event_lst)

        ray_data = {}
        ray_data["ray_events"] = merged_ray_event
        ray_data["sray"] = sray
        all_ray_events.append(ray_data)
    return all_ray_events


def create_row_column_montange(
    image_lst,
    shape=(6, 5),
    padding=30,
):

    num_row = shape[0]
    num_cols = shape[1]

    row_montage_lst = []
    for rx in range(num_row):
        row_montage = create_montage(
            image_lst[rx * num_cols : (rx + 1) * num_cols],
            stack="column",
            padding=padding,
        )
        row_montage_lst.append(row_montage)

    montage = create_montage(row_montage_lst, stack="row", padding=padding)
    return montage


def create_montage(images, stack="column", padding=30):
    img_size = images[0].shape[0:2]
    num_imgs = len(images)
    if stack == "column":
        montages = np.zeros(
            (img_size[0], img_size[1] * num_imgs + padding * (num_imgs - 1), 3)
        ).astype(np.uint8)
    else:
        montages = np.zeros(
            (img_size[0] * num_imgs + padding * (num_imgs - 1), img_size[1], 3)
        ).astype(np.uint8)

    for i in range(num_imgs):
        if stack == "column":
            montages[
                :,
                i * (img_size[1] + padding) : (i + 1) * (img_size[1]) + i * padding,
                :,
            ] = images[i]
        else:
            montages[
                i * (img_size[0] + padding) : (i + 1) * (img_size[0]) + i * padding,
                :,
                :,
            ] = images[i]
    return montages


def visualize_raw_ray_events(ray_pts, ray_event_lst, rgbd_obs_lst):
    montage_lst = []
    for px, posed_data in enumerate(rgbd_obs_lst):
        RT = posed_data["RT"]
        Kndc = posed_data["kNDC"]
        depth_img = posed_data["depth_hr"]
        image = posed_data["img"]
        ray_events = ray_event_lst[px]
        ray_img = project_points_on_image(ray_pts, (image * 255), RT, Kndc, alpha=0.8)
        depth_img = depth_image_utils.convert_depth_image_to_colormap(
            posed_data["depth"], max_depth=6
        )
        ray_img_depth = project_points_on_image(ray_pts, depth_img, RT, Kndc, alpha=0.8)

        ray_img = draw_events_on_image(ray_pts, ray_events, ray_img, RT, Kndc)
        ray_img_depth = draw_events_on_image(
            ray_pts, ray_events, ray_img_depth, RT, Kndc
        )
        padding = 20
        montage = create_montage(
            [ray_img, ray_img_depth], stack="column", padding=padding
        )
        montage_lst.append(montage)
    return montage_lst


def draw_events_on_image(ray_pts, events, image, RT, kNDC):
    event_int = (255, 0, 0)
    event_occ = (0, 0, 255)
    pts_ndc = geometry_utils.covert_world_points_to_pixel(
        ray_pts.transpose(), RT, kNDC
    ).transpose(1, 0)
    for event in events:
        event_color = (
            event_int if event.event_type == RayEventType.INTERSECTION else event_occ
        )
        index = event.index
        image = image_utils.draw_points_on_img(
            image, pts_ndc[index, None, 0:2], color=event_color
        )

    return image


def project_points_on_image(ray, image, RT, kNDC, alpha):
    pts_ndc = geometry_utils.covert_world_points_to_pixel(ray.transpose(), RT, kNDC)
    new_image = image_utils.draw_points_on_img(
        image, pts_ndc.transpose(1, 0)[:, 0:2], color=(0, 255, 0), alpha=alpha
    )
    return new_image


def convert_ray2segments(ray_signature: RaySignature):
    segments = []
    idx = 0
    num_events = len(ray_signature)
    while idx < len(ray_signature):
        event = ray_signature[idx]
        if (idx + 1) < num_events:
            next_event = ray_signature[idx + 1]
            if (event.direction == RayFreeSpaceDirection.RIGHT) and (
                next_event.direction == RayFreeSpaceDirection.LEFT
            ):
                segment = RaySignature()
                if event.index < next_event.index:
                    segment.add_event(event)
                    segment.add_event(next_event)
                    segment.evidence_count = 1
                    segments.append(segment)
                idx += 2
            else:
                idx += 1
        else:
            break

    return segments


def merge_all_ray_events(ray_signature_lst: List[RaySignature]):
    filtered_ray_signatures = []
    for ray_signature in ray_signature_lst:
        # print(ray_signature)
        if ray_signature.empty_ray():
            continue
        else:
            ray_signature.sort_events()
            filtered_ray_signatures.append(ray_signature)

    intersection_events = []

    for rx, ray_signature in enumerate(ray_signature_lst):
        for event in ray_signature:
            if event.event_type == RayEventType.INTERSECTION:
                intersection_events.append(event)
    intersection_events.sort(key=lambda x: x.index)

    # print([str(e) for e in intersection_events])
    window = 10
    cluster_lst = []
    current_cluster = []
    current_cluster_index = None
    for int_event in intersection_events:
        if len(current_cluster) == 0:
            current_cluster.append(int_event)
            current_cluster_index = int_event.index
        else:
            if abs(int_event.index - current_cluster_index) < window:
                current_cluster.append(int_event)
                current_cluster_index = np.mean([e.index for e in current_cluster])
            else:
                cluster_lst.append(current_cluster)
                current_cluster = [int_event]
                current_cluster_index = int_event.index

    if len(current_cluster) > 0:
        cluster_lst.append(current_cluster)

    ## now seperate if the clusters has two events with opposite free space direction
    new_clusters = []
    for cluster in cluster_lst:
        for current_dir in [RayFreeSpaceDirection.LEFT, RayFreeSpaceDirection.RIGHT]:
            current_cluster = []
            for event in cluster:
                if current_dir == event.direction:
                    current_cluster.append(event)
            if len(current_cluster) > 0:
                new_clusters.append(current_cluster)

    for cluster in new_clusters:
        new_int = int(np.mean([e.index for e in cluster]))
        for event in cluster:
            event.index = new_int

    # for ray_singature in filtered_ray_signatures:
    #     print(ray_singature)

    all_segments = []
    num_sigs = len(filtered_ray_signatures)
    for ix in range(num_sigs):
        segments = convert_ray2segments(filtered_ray_signatures[ix])
        if len(segments) > 0:
            all_segments.extend(segments)

    ray_signature = merge_all_segments(all_segments)
    return ray_signature


def clean_duplicate_segments(segments: List[RaySignature]):
    new_segments = []
    for segment in segments:
        if segment not in new_segments:
            new_segments.append(segment)
        else:
            index = new_segments.index(segment)
            new_segments[index].evidence_count += 1
    return new_segments


def merge_all_segments_helper(segments):
    ix = 0
    reduced_segments = []
    merged_segment = None
    while ix < len(segments):
        # breakpoint()
        if merged_segment is None:
            merged_segment = segments[ix]
        else:
            merge_possible = possible_to_merge(merged_segment, segments[ix])
            if merge_possible:
                old_merged_segment = merged_segment
                merged_segment, force_merged = merge_two_segments(
                    merged_segment, segments[ix]
                )
                if force_merged is True:
                    # reduced_segments.append(segments[ix])
                    merged_segment = old_merged_segment
            else:
                # print('merged segment -- {}'.format(str(merged_segment)))
                reduced_segments.append(merged_segment)
                merged_segment = segments[ix]
        ix += 1
    if merged_segment is not None:
        reduced_segments.append(merged_segment)
    reduced_segments.sort(key=lambda x: x[0].index)
    return reduced_segments


def merge_all_segments(segments):
    ## sort segments by start_index

    segments.sort(key=lambda x: x[0].index)
    segments = clean_duplicate_segments(segments)
    if False:
        print("input segments")
        for seg in segments:
            print(f"{str(seg)}  evidence: {str(seg.evidence_count)}")

    reduced_segments = []
    merged_segment = None
    ix = 0
    # if True:
    #     for seg in segments:
    #         print(seg)

    num_segs = len(segments)
    segment_conflict_matrix = np.zeros((num_segs, num_segs))
    mergable_segments = np.zeros((num_segs, num_segs))
    for ix in range(num_segs):
        for jx in range(ix, num_segs):
            merge_possible = possible_to_merge(segments[ix], segments[jx])
            # if ix == 5 and jx == 9:
            #     pdb.set_trace()
            if merge_possible:
                mergable_segments[ix, jx] = 1
                merged_segment, forced_merge = merge_two_segments(
                    segments[ix], segments[jx]
                )
                if forced_merge:
                    segment_conflict_matrix[ix, jx] = 1
    segment_conflict_matrix += segment_conflict_matrix.T
    seg_conflicts = np.sum(segment_conflict_matrix, axis=1)
    # print(seg_conflicts)
    sorted_inds = np.argsort(seg_conflicts, kind="mergesort")
    sorted_segments = []
    for ix in sorted_inds:
        sorted_segments.append(segments[ix])

    reduced_segments = merge_all_segments_helper(segments)
    if False:
        print("reduced segments")
        for seg in reduced_segments:
            print(f"{str(seg)}  evidence: {str(seg.evidence_count)}")
        print("-----------------")
    # ray_signature = RaySignature()
    # for seg in reduced_segments:
    #     ray_signature.add_segment(seg)
    # ray_signature.sort_events()
    # pdb.set_trace()
    return reduced_segments


def possible_to_merge(seg1: RaySignature, seg2: RaySignature):
    ## sees if these two segments can be merged into 1.

    if seg1[0].index > seg2[0].index:
        seg1, seg2 = seg2, seg1

    start1 = seg1[0]
    end1 = seg1[1]

    start2 = seg2[0]
    end2 = seg2[1]

    no_overlap = end1.index <= start2.index
    return not no_overlap


occlusion_evnts = [
    RayEventType.OCCLUSION,
    RayEventType.START,
    RayEventType.DISOCCLUSION,
]


def merge_two_events(event1, event2):

    if event1.direction == RayFreeSpaceDirection.RIGHT:
        ## this in case of start events. event1 starts before event2
        if event1.index > event2.index:
            event1, event2 = event2, event1
    else:
        ## this is in case of end events
        ## event 1 ends after event 2
        if event1.index < event2.index:
            event1, event2 = event2, event1

    if event1.event_type in occlusion_evnts and event2.event_type in occlusion_evnts:
        if event1.direction == RayFreeSpaceDirection.RIGHT:
            return event1 if event1.index < event2.index else event2
        else:
            return event1 if event1.index > event2.index else event2
    elif event1.event_type == event2.event_type == RayEventType.INTERSECTION:
        if event1.index == event2.index and event1.direction == event2.direction:
            return event1
        else:
            return None
    elif (
        event1.event_type == RayEventType.INTERSECTION
        or event2.event_type == RayEventType.INTERSECTION
    ):
        ## only one of them is an intersection event
        event = event1 if event1.event_type == RayEventType.INTERSECTION else event2
        return event
    else:
        return None


def merge_in_conflict(seg1: RaySignature, seg2: RaySignature):
    if seg1[0].index > seg2[0].index:
        seg1, seg2 = seg2, seg1

    new_start = merge_two_events(seg1[0], seg2[0])
    new_end = merge_two_events(seg1[1], seg2[1])

    valid_1 = False
    valid_2 = False
    if new_start is None:
        ## do something
        pdb.set_trace()
        assert False, "how to merge?"
    elif new_end is None:
        if seg1[1].event_type == seg2[1].event_type == RayEventType.INTERSECTION:
            if seg1[1].index < seg2[1].index:
                new_end = seg1[1]
                valid_1 = True
            elif seg2[1].index < seg1[1].index:
                new_end = seg2[1]
                valid_2 = True
        else:
            pdb.set_trace()
            assert False, "how to merge?"

    new_segment = RaySignature([new_start, new_end])
    new_segment.evidence_count = seg1.evidence_count if valid_1 else seg2.evidence_count
    return new_segment


def merge_two_segments(seg1: RaySignature, seg2: RaySignature):
    ## merges two segments into one.
    if seg1[0].index > seg2[0].index:
        seg1, seg2 = seg2, seg1

    new_start = merge_two_events(seg1[0], seg2[0])
    new_end = merge_two_events(seg1[1], seg2[1])
    force_merge = False
    if new_start is None or new_end is None:
        force_merge = True
        new_segment = None
        # new_segment = merge_in_conflict(seg1, seg2)
    else:
        force_merge = False
        evidence_count = seg1.evidence_count + seg2.evidence_count
        new_segment = RaySignature([new_start, new_end])
        new_segment.evidence_count = evidence_count
    if new_segment is None and not (force_merge):
        pdb.set_trace()
    return new_segment, force_merge


def check_valid_merge(segment):
    if len(segment) % 2 == 0:
        valid_merge = True

        for ix in range(0, len(segment), 2):
            valid_merge = (
                valid_merge
                and (segment[ix].direction == RayFreeSpaceDirection.RIGHT)
                and (segment[ix + 1].direction == RayFreeSpaceDirection.LEFT)
                and (segment[ix + 1].index > segment[ix].index)
            )
        return valid_merge
    else:
        return False


def get_opposite_direction(direction: RayFreeSpaceDirection):
    if direction == RayFreeSpaceDirection.LEFT:
        return RayFreeSpaceDirection.RIGHT
    else:
        return RayFreeSpaceDirection.LEFT


def compute_distance_function(segment, og_points, sampled_pts, window=0.2):
    start_event = segment[0]
    end_event = segment[1]
    start_pts = og_points[start_event.index]
    end_pts = og_points[end_event.index]
    segment_pts = np.array([start_pts, end_pts])
    distance2events = np.linalg.norm(
        sampled_pts[:, None, :] - segment_pts[None], axis=-1
    )

    closer = np.argmin(distance2events, axis=-1)
    segment_len = np.linalg.norm(segment_pts[1] - segment_pts[0])
    ## check which points are in the segment?
    validity = np.abs(np.sum(distance2events, axis=1) - segment_len) < 1e-3
    start_type = start_event.event_type
    end_type = end_event.event_type
    penalty_region_values = sampled_pts * 0  ## boundaries
    penalty_types = (sampled_pts[:, 0] * 0).astype(
        int
    )  ## types of penalty -- force the actual distance to be less than, equal or greater than.
    below = distance2events[:, 0]
    above = distance2events[:, 1]
    penalty_region_values[:, 0] = -1 * below
    penalty_region_values[:, 1] = 1 * above
    if (
        start_type == RayEventType.INTERSECTION
        and end_type == RayEventType.INTERSECTION
    ):
        penalty_types[:] = 1  ## II
    elif start_type in occlusion_evnts and end_type in occlusion_evnts:
        penalty_types[:] = 2  ## OO
    elif start_type == RayEventType.INTERSECTION and end_type in occlusion_evnts:
        penalty_types[:] = 3  ## IO
    elif start_type in occlusion_evnts and end_type == RayEventType.INTERSECTION:
        if start_event.index < 10:
            penalty_types[:] = 5
        else:
            penalty_types[:] = 4  ## OI
    else:
        pdb.set_trace()
        assert False

    if start_type == RayEventType.INTERSECTION:

        extended_validity = np.logical_and(
            distance2events[:, 0] < window,
            np.sum(distance2events, axis=1) > segment_len,
        )

        validity = np.logical_or(validity, extended_validity)
        penalty_types[extended_validity] = 1
        penalty_region_values[extended_validity, 0] = below[extended_validity]

    if end_type == RayEventType.INTERSECTION:

        extended_validity = np.logical_and(
            distance2events[:, 1] < window,
            np.sum(distance2events, axis=1) >= segment_len + 1e-4,
        )
        validity = np.logical_or(validity, extended_validity)
        penalty_types[extended_validity] = 1
        penalty_region_values[extended_validity, 1] = -1 * above[extended_validity]

    return penalty_region_values, penalty_types, validity


def sample_points_segments(og_points, ray_direction, segment, num_samples, std):
    start_event = segment[0]
    end_event = segment[1]
    og_point_start = og_points[start_event.index]
    og_point_end = og_points[end_event.index]

    uniform_lambda = np.random.uniform(0, 1, size=num_samples)
    uniform_points = (
        og_point_start[None, :]
        + uniform_lambda[:, None] * (og_point_end - og_point_start)[None, :]
    )

    direction = ray_direction[start_event.index]
    normal_points = []
    if start_event.event_type == RayEventType.INTERSECTION:
        pts = sample_points_near_intersection(
            start_event, og_point_start, direction, num_samples=num_samples, std=std
        )
        normal_points.append(pts)
    if end_event.event_type == RayEventType.INTERSECTION:
        pts = sample_points_near_intersection(
            start_event, og_point_start, direction, num_samples=num_samples, std=std
        )
        normal_points.append(pts)
    if len(normal_points) > 0:
        normal_points = np.concatenate(normal_points, axis=0)
        points = np.concatenate([uniform_points, normal_points], axis=0)
    else:
        points = uniform_points
    direction = points * 0 + direction[None, :]
    points = np.concatenate([points, direction], axis=-1)
    return points


def sample_points_near_intersection(event, point, direction, num_samples, std=0.1):
    normal_lambda = np.random.normal(0, 0.1, num_samples)
    points = point[None, :] + normal_lambda[:, None] * direction[None, :]
    return points


def sample_points_from_ray_signatures(
    ray_events_lst, ref_img_data, intersection_thickness=20
):
    all_points = []
    ray_distance_lst = []
    for ray_events in ray_events_lst:
        og_points = ray_events["sray"]
        ray_segments = ray_events["ray_events"]
        if False:
            sampled_pts = ray_events["sray"] * 1
        # breakpoint()
        if True:
            sampled_pts = []
            if len(ray_segments) > 0:
                for segment in ray_segments:
                    pts = sample_points_segments(
                        og_points[:, 0:3],
                        og_points[:, 3:6],
                        segment,
                        num_samples=512,
                        std=0.1,
                    )
                    sampled_pts.append(pts)
            if len(sampled_pts) > 0:
                sampled_pts = np.concatenate(sampled_pts, axis=0)
            else:
                continue

        ray_distance = {
            "penalty_regions": sampled_pts[:, 0:3] * 0,
            "validity": sampled_pts[:, 0] * 0,
            "penalty_types": sampled_pts[:, 0] * 0,
        }  ## :0 -- distnace, :1 -- validity, :2 -- distance type
        all_points.append(sampled_pts)
        # if len(ray_segments) >= 2:
        #     breakpoint()
        for segment in ray_segments:
            penalty_regions, penalty_types, validity = compute_distance_function(
                segment, og_points[:, 0:3], sampled_pts[:, 0:3]
            )
            # cat_sig = np.stack([penalty_regions, penalty_types, validity * 1], axis=1)
            ## this combining works as the events are non-overlapping
            ray_distance["penalty_regions"][validity] = penalty_regions[validity]
            ray_distance["validity"][validity] = validity[validity] * 1
            ray_distance["penalty_types"][validity] = penalty_types[validity]
        ray_distance_lst.append(ray_distance)
    ray_distance_all = {}
    if len(ray_distance_lst) > 0:
        for key in ray_distance_lst[0].keys():
            ray_distance_all[key] = np.concatenate([k[key] for k in ray_distance_lst])
        all_points = np.concatenate(all_points)
        return all_points, ray_distance_all
    else:
        return None, None
