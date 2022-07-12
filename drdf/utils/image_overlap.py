import pdb

import numpy as np
import torch

from . import geometry_utils, grid_utils
from . import image as image_utils
from . import tensor_utils


def color_occluded_regions(ref_data, aux_data):
    """
    Create a color image of the occluded regions in the reference image -- coloring is on the aux image.
    ref_data: dict with keys: 'depth', 'RT', 'kNDC', 'img'
    aux_data: dict with keys 'depth', 'RT', 'kNDC', 'img'
    """

    ref_depth = ref_data["depth"]
    ref_img = ref_data["img"]
    ref_RT = ref_data["RT"]
    ref_kNDC = ref_data["kNDC"]

    aux_depth = aux_data["depth"]
    aux_img = aux_data["img"]
    aux_RT = aux_data["RT"]
    aux_kNDC = aux_data["kNDC"]
    use_cuda = False
    points_aux, valid_points_aux, ndc_pts = geometry_utils.convert_depth_pcl(
        aux_depth, aux_RT, aux_kNDC, use_cuda=use_cuda, return_xyz=True
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

    ref_depth = tensor_utils.tensor_to_cuda(ref_depth, cuda=use_cuda)
    points_ref_depth = image_utils.interpolate_depth(
        ref_depth[None], points_ref[None, :, 0:2]
    )[0, 0]

    valid_ref_depth = points_ref_depth > 0.1
    all_valid = valid_ref * valid_ref_depth * valid_points_aux

    occluded_pts = (points_ref_depth - points_ref[:, 2]) < -0.05
    valid_occluded = (occluded_pts * all_valid).type(torch.bool)

    occluded_locations = ndc_pts[valid_occluded == True]
    colored_img = color_image(aux_img, occluded_locations[:, 0:2])
    return colored_img


def color_image(image, ndc_pixels):
    image = image * 1

    new_img = image_utils.draw_points_on_img(
        image, ndc_pixels, draw_index=False, color=(255, 0, 0), alpha=0.5
    )
    return new_img
