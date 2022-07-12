import pdb
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
from cv2 import merge

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


class RaySignature:
    def __init__(self, events: List[RayEvent] = []):
        if len(events) == 0:
            self.events = []
        else:
            self.events = [k for k in events]

    def add_event(self, ray_event: RayEvent):
        self.events.append(ray_event)

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
        frustum_clip_locations[1], RayFreeSpaceDirection.LEFT, RayEventType.END
    )
    # pdb.set_trace()
    # ray_signature.add_event(start_event)
    # ray_signature.add_event(end_event)

    ray_distances_non_nan = pts_z - pts_depths
    ray_distances = pts_z - pts_depths

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
        ray_distances, window=window, return_sign=True
    )
    discontunities = detect_discontunity(ray_distances, window=1)
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

            ray_signature.add_event(end_event)

    # ray_signature.clean_events()
    # pdb.set_trace()
    return ray_signature


def fill_missing_nans(ray_dists, window=10):

    nan_inds = np.where(np.isnan(ray_dists))[0]
    components = []
    current_ind = None
    current_cluster = []
    cluster_lables = []
    cluster_ind = None
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


def create_ray_event(event_loc, ray_distances):

    return


def valid_depths(depths):
    return depths > 1e-3


def detect_discontunity(ray_dist, window):
    length = len(ray_dist)
    ray_discont = np.abs(ray_dist[:-1] - ray_dist[1:]) > 0.4
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
    ray_events = get_ray_events(
        points, RT, pts_z, pts_depth, pts_validity, depth_validity, window=window
    )
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


def compute_all_ray_signatures(sampled_rays, ref_imgdata, nbr_posed_img_data):

    num_nbrs = len(nbr_posed_img_data)

    rgbd_obs_lst = [ref_imgdata] + nbr_posed_img_data

    num_obs = len(rgbd_obs_lst)

    all_ray_events = []
    aggregrated_ray_data = []
    for sx, sray in enumerate(sampled_rays[18:]):
        ray_event_lst = []
        sray_points = sray[..., 0:3]
        for px, posed_data in enumerate(rgbd_obs_lst):
            RT = posed_data["RT"]
            Kndc = posed_data["kNDC"]
            depth_img = posed_data["depth_hr"]
            # if px == 14:
            #     pdb.set_trace()

            ray_events = compute_ray_signature(sray_points, RT, Kndc, depth_img)
            # print('posed ray sig {}'.format(px))
            # print(ray_events)
            ray_event_lst.append(ray_events)

        merged_ray_event = merge_all_ray_events(ray_event_lst)
        all_ray_events.append(merged_ray_event)
    pdb.set_trace()
    return all_ray_events


def merge_all_ray_events(ray_singature_lst: List[RaySignature]):
    filtered_ray_signatures = []

    for ray_signature in ray_singature_lst:
        # print(ray_signature)
        if ray_signature.empty_ray():
            continue
        else:

            filtered_ray_signatures.append(ray_signature)
        # print(ray_signature)

    intersection_events = []

    for rx, ray_signature in enumerate(ray_singature_lst):
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

    for ray_singature in filtered_ray_signatures:
        print(ray_singature)

    num_sigs = len(filtered_ray_signatures)
    merged_ray = RaySignature()
    print("---- Merged Rays ----")
    for ix in range(num_sigs):
        # print('---- {} '.format(ix))
        # print(merged_ray)
        merged_ray = merge_two_ray(merged_ray, filtered_ray_signatures[ix])
    print(merged_ray)

    return merged_ray


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
                segment.add_event(event)
                segment.add_event(next_event)
                segments.append(segment)
                idx += 2
            else:
                idx += 1
        else:
            break
    return segments


def overlap_bw_segments(segment1, segment2):
    s1_0 = segment1[0].index
    s1_1 = segment1[1].index

    s2_0 = segment2[0].index
    s2_1 = segment2[1].index

    if (s1_1 < s2_0) and (s1_1 < s2_1):
        return False
    elif (s1_0 > s2_0) and (s1_0 > s2_1):
        return False
    else:
        return True


def intersecting_segments(segment1, segment2):

    if segment1[0].index > segment2[0].index:
        segment1, segment2 = segment2, segment1

    s1_0 = segment1[0].index
    s1_1 = segment1[1].index

    s2_0 = segment2[0].index
    s2_1 = segment2[1].index

    if (s1_0 < s2_0) and (s1_1 < s2_1) and (s1_1 >= s2_0):
        return True
    return False


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


# def union_two_segments(segment1, segment2):
#     occlusion_evnts = [
#         RayEventType.OCCLUSION,
#         RayEventType.START,
#         RayEventType.DISOCCLUSION,
#     ]
#     if segment1[0].index > segment2[0].index:
#         segment1, segment2 = segment2, segment1

#     start1 = segment1[0]
#     start2 = segment2[0]

#     end1 = segment1[1]
#     end2 = segment2[1]
#     merged_events = []

#     merged_events.append(start1)

#     if end1.index <= start2.index:
#         ## end1 is before the start2 of 2 --almost not overlap
#         if end1

#     elif end2.index > start2.index:


def union_two_segments(segment1, segment2):

    occlusion_evnts = [
        RayEventType.OCCLUSION,
        RayEventType.START,
        RayEventType.DISOCCLUSION,
    ]

    if segment1[0].index > segment2[0].index:
        segment1, segment2 = segment2, segment1

    start1 = segment1[0]
    start2 = segment2[0]

    end1 = segment1[1]
    end2 = segment2[1]
    # pdb.set_trace()
    merged_events = []
    if (
        (end1.index == start2.index)
        and (end1.direction == RayFreeSpaceDirection.LEFT)
        and (start2.direction == RayFreeSpaceDirection.RIGHT)
        and (end1.event_type not in occlusion_evnts)
    ):
        merged_events.append(start1)

        if end1.event_type == start2.event_type == RayEventType.INTERSECTION:
            merged_events.append(end1)
            merged_events.append(start2)
        merged_events.append(end2)
    elif start1.event_type in occlusion_evnts and start2.event_type in occlusion_evnts:
        merged_events.append(start1)
        direction = start1.direction
        if end1.event_type == RayEventType.INTERSECTION:
            if end1.index < end2.index:
                merged_events.append(end1)
                if (
                    end2.event_type in occlusion_evnts
                    or end2.event_type == RayEventType.INTERSECTION
                    or end2.event_type == RayEventType.END
                ) and end2.direction == RayFreeSpaceDirection.LEFT:
                    new_event = RayEvent(
                        end1.index,
                        RayFreeSpaceDirection.RIGHT,
                        RayEventType.INTERSECTION,
                    )
                    merged_events.append(new_event)
                    merged_events.append(end2)
            else:
                pdb.set_trace()
                assert False, "not implemented"
        elif end2.event_type == RayEventType.INTERSECTION:
            if end1.index < end2.index:
                if end1.event_type in occlusion_evnts:
                    merged_events.append(end2)
                else:
                    pdb.set_trace()
                    assert False, "not implemented"
            elif end1.index >= end2.index:
                pdb.set_trace()
                if (
                    end1.event_type in occlusion_evnts
                    or end1.event_type == RayEventType.INTERSECTION
                    or end1.event_type == RayEventType.END
                ) and end1.direction == RayFreeSpaceDirection.LEFT:
                    merged_events.append(end2)
                    new_event = RayEvent(
                        end2.index,
                        RayFreeSpaceDirection.RIGHT,
                        RayEventType.INTERSECTION,
                    )
                    merged_events.append(new_event)
                    merged_events.append(end1)
                else:
                    pdb.set_trace()
                    assert False, "not implemented"
        elif end1.event_type in occlusion_evnts and end2.event_type in occlusion_evnts:
            end_event = end1 if end1.index > end2.index else end2
            merged_events.append(end_event)
        elif end1.event_type == RayEventType.END:
            merged_events.append(end2)
        elif end2.event_type == RayEventType.END:
            merged_events.append(end1)
        else:
            pdb.set_trace()
            assert False, "not implemented"

    elif start1.event_type == RayEventType.INTERSECTION:
        merged_events.append(start1)
        intersect_end = None
        if start2.event_type in occlusion_evnts:
            if start2.index == end1.index:
                if end1.event_type in occlusion_evnts:
                    end_event = None
                else:
                    pdb.set_trace()
                    assert False, "already handled above"
            elif start2.index < end1.index:
                end_event = None
            else:
                pdb.set_trace()
                assert False, "will never happen, other wise there is no intersection"
                end_event = end1
        elif start2.event_type == RayEventType.INTERSECTION:
            if start2.index < end1.index:
                new_event = RayEvent(
                    start2.index, RayFreeSpaceDirection.LEFT, start2.event_type
                )
                merged_events.append(new_event)
                merged_events.append(start2)
                pdb.set_trace()
            elif start2.index == end1.index:
                if start2.event_type == end1.event_type:
                    merged_events.append(end1)
                    merged_events.append(start2)
            else:
                pdb.set_trace()
                assert False, "not implemented"
        else:
            ## this means it is an intersection event
            pdb.set_trace()
            assert False, "see a case here"

        if end2.event_type == RayEventType.INTERSECTION:
            if end1.index < end2.index:
                merged_events.append(end2)
            else:
                pdb.set_trace()
                assert False, "not implemented"
        elif end1.event_type == RayEventType.INTERSECTION:
            pdb.set_trace()
            if end1.index <= end2.index and (start2.index < end1.index):
                merged_events.append(end1)

        else:
            if end1.index < end2.index:
                merged_events.append(end2)
            else:
                merged_events.append(end1)
    else:
        pdb.set_trace()
        assert False, "not implemented"

    merged_signature = RaySignature()
    for event in merged_events:
        merged_signature.add_event(event)
    return merged_signature


def find_inteserction_of_sequences(seq1, seq2):
    start1, end1 = seq1[0], seq1[1]
    start2, end2 = seq2[0], seq2[1]
    min_ind = min(start1, start2)
    max_ind = max(end1, end2)
    free_space = np.array([False for i in range(min_ind, max_ind + 1)])
    free_space[start1 + 1 - min_ind : end1 - 1 - min_ind] = True
    free_space[start2 + 1 - min_ind : end2 - 1 - min_ind] = True
    free_inds = np.where(free_space == True)[0]
    start = free_inds[0] + min_ind
    end = free_inds[-1] + min_ind
    return start, end


## Only used when one segment is subset of the the other segment
def merge_two_segments(segment1, segment2):

    if segment1[0].index > segment2[0].index:
        segment1, segment2 = segment2, segment1

    start1 = segment1[0]
    start2 = segment2[0]

    end1 = segment1[1]
    end2 = segment2[1]

    s1_0 = segment1[0].index
    s1_1 = segment1[1].index

    s2_0 = segment2[0].index
    s2_1 = segment2[1].index

    merged_events = []
    start_event = None

    occlusion_evnts = [
        RayEventType.OCCLUSION,
        RayEventType.START,
        RayEventType.DISOCCLUSION,
    ]
    if s1_0 < s2_0:
        if start2.event_type in occlusion_evnts:
            if (
                start1.event_type in occlusion_evnts
                or start1.event_type == RayEventType.INTERSECTION
            ):
                start_event = start1
                merged_events.append(start_event)
        elif start2.event_type == RayEventType.INTERSECTION:
            assert (
                start2.index < end1.index
            ), "why is this part of merge segemnts -- should go to union"
            merged_events.append(start1)
            new_event = RayEvent(
                start2.index, RayFreeSpaceDirection.LEFT, RayEventType.INTERSECTION
            )
            merged_events.append(new_event)
            merged_events.append(start2)
    elif s2_0 == s1_0:
        if (
            segment1[0].event_type in occlusion_evnts
            and segment2[0].event_type in occlusion_evnts
        ):
            start_event = segment1[0]
            merged_events.append(start_event)
        elif (
            segment1[0].event_type
            == segment2[0].event_type
            == RayEventType.INTERSECTION
        ):
            start_event = segment1[0]
            merged_events.append(start_event)
        elif (segment1[0].event_type == RayEventType.INTERSECTION) or (
            segment2[0].event_type == RayEventType.INTERSECTION
        ):
            start_event = (
                start1 if start1.event_type == RayEventType.INTERSECTION else start2
            )
            if start1.direction == start2.direction:
                merged_events.append(start_event)
            else:
                assert False, "something is wrong -- maybe check again"
    else:
        assert False, "how is this an overlap? -- they are equal?"

    if len(merged_events) == 0:
        pdb.set_trace()
        assert False, "how is start_event none?"
    if end1.index == end2.index:
        ## (0, RayEventType.START, -->) (287, RayEventType.INTERSECTION, <--)
        ## (150, RayEventType.DISOCCLUSION, -->) (287, RayEventType.OCCLUSION, <--)
        ##
        if segment1[1].direction == segment2[1].direction:
            if (segment1[1].event_type == RayEventType.INTERSECTION) or (
                segment2[1].event_type == RayEventType.INTERSECTION
            ):
                end_event = (
                    segment1[1]
                    if segment1[1].event_type == RayEventType.INTERSECTION
                    else segment2[1]
                )
                merged_events.append(end_event)
            else:
                merged_events.append(segment1[1])
        else:
            pdb.set_trace()
            assert False, "waiting for other cases"

    elif end1.index < end2.index:
        if segment1[1].event_type == RayEventType.INTERSECTION:
            end_event = segment1[1]
            merged_events.append(end_event)
            new_event = RayEvent(
                end_event.index, RayFreeSpaceDirection.RIGHT, RayEventType.INTERSECTION
            )
            merged_events.append(new_event)
            merged_events.append(segment2[1])
            assert segment2[1].direction == RayFreeSpaceDirection.LEFT, "cannot happen"
        elif end1.event_type in occlusion_evnts:
            if end2.event_type == RayEventType.END:
                end_event = end1
                merged_events.append(end_event)
            else:
                end_event = end2
                merged_events.append(end_event)
        elif segment1[1].event_type == RayEventType.END:
            end_event = segment2[1]
            merged_events.append(end_event)

    elif end2.index < end1.index:
        if segment2[1].event_type == RayEventType.INTERSECTION:
            end_event = segment2[1]
            merged_events.append(end_event)
            new_event = RayEvent(
                end_event.index, RayFreeSpaceDirection.RIGHT, RayEventType.INTERSECTION
            )
            merged_events.append(new_event)
            merged_events.append(segment1[1])
        elif segment2[1].event_type in occlusion_evnts:
            if segment1[1].event_type == RayEventType.END:
                end_event = segment2[1]
                merged_events.append(
                    end_event
                )  ## this is because end event has much less information than the occlusion event.
            else:
                end_event = segment1[1]
                merged_events.append(end_event)
        elif segment2[1].event_type == RayEventType.END:
            end_event = segment1[1]
            merged_events.append(end_event)
            pdb.set_trace()
            assert False, "how is it here"
    else:
        assert False, "how is this happening??"

    merged_segment = RaySignature()
    for event in merged_events:
        merged_segment.add_event(event)
    return merged_segment


def merge_two_ray(ray_signature_1, ray_signature_2):
    ray_signature_1.sort_events()
    ray_signature_2.sort_events()

    segments1_lst = convert_ray2segments(ray_signature_1)
    segments2_lst = convert_ray2segments(ray_signature_2)

    if False:
        print("Ray 1 segments:")
        for seg in segments1_lst:
            print(f"{seg}")

        print("Ray 2 segments:")
        for seg in segments2_lst:
            print(f"{seg}")

    # for seg in segments2_lst:
    #     print('{}  {} '.format(seg[0], seg[1]))

    s1_idx = 0
    s2_idx = 0
    final_segments = []
    # pdb.set_trace()
    if len(segments2_lst) >= 1 and len(segments1_lst) >= 1:
        seg1 = segments1_lst[s1_idx]
        seg2 = segments2_lst[s2_idx]
        while True:
            s1_merged = False
            s2_merged = False
            break_flag = False
            if len(seg1) > 2 or len(seg2) > 2:
                pdb.set_trace()
            overlap = overlap_bw_segments(seg1, seg2)

            merged_segment = None

            if overlap:

                intersect = intersecting_segments(seg1, seg2)
                if intersect:
                    merged_segment = union_two_segments(seg1, seg2)
                else:
                    merged_segment = merge_two_segments(seg1, seg2)
                valid_merge = check_valid_merge(merged_segment)

                if not valid_merge:
                    pdb.set_trace()
                # if seg[1].index == seg2[1].index:
                #     s1_merged = True
                #     s2_merged = True
                elif seg1[1].index > seg2[1].index:
                    s1_merged = True
                    seg1 = merged_segment
                    s2_idx += 1
                else:
                    s2_merged = True
                    seg2 = merged_segment
                    s1_idx += 1
            else:
                if seg1[0].index < seg2[0].index:
                    final_segments.append(seg1)
                    s1_idx += 1

                else:
                    final_segments.append(seg2)
                    s2_idx += 1

            if not s1_merged:
                if s1_idx < len(segments1_lst):
                    seg1 = segments1_lst[s1_idx]
                    if merged_segment is not None:
                        if len(seg2) > 2:
                            assert len(seg2) % 2 == 0, "len seg2 is not even"
                            final_segments.append(RaySignature(seg2[:-2]))
                            seg2 = RaySignature(seg2[-2:])
                            merged_segment = seg2

                else:
                    if not s2_merged:
                        final_segments.append(seg2)
                    for s2_ix in range(s2_idx + 1, len(segments2_lst)):
                        final_segments.append(segments2_lst[s2_ix])
                    break_flag = True
            if not s2_merged:
                if s2_idx < len(segments2_lst):
                    seg2 = segments2_lst[s2_idx]
                    if merged_segment is not None:
                        if len(seg1) > 2:
                            assert len(seg1) % 2 == 0, "len seg2 is not even"
                            final_segments.append(RaySignature(seg1[:-2]))
                            seg1 = RaySignature(seg1[-2:])
                            merged_segment = seg1

                else:
                    if not s1_merged:
                        final_segments.append(seg1)
                    for s1_ix in range(s1_idx + 1, len(segments1_lst)):
                        final_segments.append(segments1_lst[s1_ix])
                    break_flag = True

            if break_flag:
                if merged_segment is not None:
                    final_segments.append(merged_segment)
                break
            # else:
            #     pdb.set_trace()
            # elif (s1_merged or s2_merged):
            #     pdb.set_trace()
            #     if merged_segment is not None:
            #         if(len(merged_segment) >= 1):
            #             final_segments.append(merged_segment)
            #         final_segments.append(merged_segment)
            # pdb.set_trace()

    elif len(segments2_lst) >= 1:
        final_segments = segments2_lst
    elif len(segments1_lst) >= 1:
        final_segments = segments1_lst

    ray_signature = RaySignature()
    for segment in final_segments:
        for evnt in segment:
            ray_signature.add_event(evnt)
    ray_signature.sort_events()

    return ray_signature

    # merged_ray = RaySignature()
    # prev_event = None
    # segments_r1 = []
    # segments_r2 = []

    # while(True):

    # while (True):
    #     r1_segment_start = ray_signature_1[r1_indx]
    #     r1_segment_end = ray_signature_1[r1_indx + 1]

    #     r2_segment_start = ray_signature_2[r2_indx]
    #     r2_segment_end = ray_signature_2[r2_indx + 1]

    #     event_r1 = ray_signature_1[r1_indx]
    #     event_r2 = ray_signature_2[r2_indx]
    #     if event_r1.index < event_r2.index:
    #         new_event = event_r1
    #         r1_indx += 1
    #     elif event_r1.index > event_r2.index:
    #         new_event = event_r2
    #         r2_indx += 1
    #     else:
    #         ## intersection event maybe -- since same index
    #         if (event_r1.event_type == RayEventType.INTERSECTION
    #             ) and (event_r2.event_type == RayEventType.INTERSECTION):
    #             new_event = event_r1

    # return


def get_opposite_direction(direction: RayFreeSpaceDirection):
    if direction == RayFreeSpaceDirection.LEFT:
        return RayFreeSpaceDirection.RIGHT
    else:
        return RayFreeSpaceDirection.LEFT
