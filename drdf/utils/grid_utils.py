import pdb

import numpy as np
import torch


def create_pixel_aligned_grid(resX, resY, resZ, b_min, b_max, transform=None):
    """
    Create a pixel aligned grid.

    """

    length = b_max - b_min
    xy_coords = np.mgrid[
        :resX,
        :resY,
    ]
    xy_coords = xy_coords.reshape(2, -1)
    xy_matrix = np.eye(3)
    coords_matrix = np.eye(4)

    coords_matrix[0, 0] = length[0] / resX
    coords_matrix[1, 1] = length[1] / resY
    coords_matrix[2, 2] = length[2] / resZ

    res = np.array([resX, resY, resZ])

    xy_matrix[0, 0] = length[0] / resX
    xy_matrix[1, 1] = length[1] / resY
    xy_matrix[0:2, 2] = b_min[
        0:2,
    ]
    xy_coords = np.matmul(xy_matrix[:2, :2], xy_coords) + xy_matrix[:2, 2:3]
    depths = np.mgrid[:resZ] * length[2] / resZ + b_min[2]
    depths = depths[None, None, :].repeat(resX, axis=0).repeat(resY, axis=1)
    xy_coords = xy_coords.reshape(2, resX, resY)
    coords = []
    for dx in range(resZ):
        coords.append(
            np.concatenate(
                [xy_coords * depths[None, :, :, dx], depths[None, :, :, dx]], axis=0
            )
        )
    coords = np.stack(coords, axis=-1)

    if transform is not None:
        coords = np.matmul(transform[:3, :3], coords) + transform[:3, 3:4]
        coords_matrix = np.matmul(transform, coords_matrix)
    coords = coords.reshape(3, resX, resY, resZ)
    coords = torch.FloatTensor(coords)
    return coords


def sample_img_grid_ndc(img_size):
    img_h, img_w = img_size[1], img_size[0]
    x = np.linspace(0, img_w - 1, num=img_h)
    y = np.linspace(0, img_h - 1, num=img_w)
    xs, ys = np.meshgrid(y, x)
    coordinates = np.stack([xs / (img_w - 1), ys / (img_h - 1)], axis=0)
    coordinates = coordinates * 2 - 1
    return coordinates
