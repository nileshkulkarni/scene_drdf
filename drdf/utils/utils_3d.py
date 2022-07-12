import pdb

import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm


def convert_to_color(value, min_val, max_val, cm):
    """
    value: (B, N)
    min_val: (B, N)
    max_val: (B, N)
    cm: colormap
    """
    value = np.clip(value, a_min=min_val, a_max=max_val)
    value = (value - min_val) / (max_val - min_val)
    # value = value * 255
    # value = value.astype(np.uint8)
    colors = cm(value)
    colors = colors[..., :3]
    return colors


def render_mesh(pixel_coords, depth, img_size, color_depth=False):
    """
    pixel_coords: 2 x N
    img_size: (H, W)
    """

    mask_img = np.zeros((img_size[0], img_size[1], 3))
    x = pixel_coords[0, :]
    y = pixel_coords[1, :]

    x = np.round(np.array(x)).astype(np.int)
    y = np.round(np.array(y)).astype(np.int)

    valid = np.logical_and(
        np.logical_and(x >= 0, x < img_size[0]), np.logical_and(y >= 0, y < img_size[1])
    )
    valid_inds = np.where(valid)[0]
    x = x[valid_inds]
    y = y[valid_inds]

    if color_depth:
        magma_cm = cm.get_cmap("magma")
        colors = convert_to_color(depth[valid_inds], min_val=0, max_val=13, cm=magma_cm)
        mask_img[y, x] = colors
    else:
        mask_img[y, x] = 1
        mask_img = mask_img[:, :, None] * np.ones((1, 1, 3))

    mask_img = (mask_img * 255).astype(np.uint8)
    return mask_img


def render_mesh_cv2(pixel_coords, depth, img_size, color_depth=False):

    mask_img = np.zeros((img_size[0], img_size[1], 3))
    x = pixel_coords[0, :]
    y = pixel_coords[1, :]

    x = np.round(np.array(x)).astype(np.int)
    y = np.round(np.array(y)).astype(np.int)

    valid = np.logical_and(
        np.logical_and(x >= 0, x < img_size[0]), np.logical_and(y >= 0, y < img_size[1])
    )
    valid_inds = np.where(valid)[0]
    x = x[valid_inds]
    y = y[valid_inds]

    if color_depth:
        magma_cm = cm.get_cmap("magma")
        colors = convert_to_color(depth[valid_inds], min_val=0, max_val=13, cm=magma_cm)

        for (xi, xj), xcolor in zip(zip(x, y), colors):
            xcolor = (xcolor * 255).astype(int)
            mask_img = cv2.circle(
                mask_img,
                (xi, xj),
                2,
                (int(xcolor[0]), int(xcolor[1]), int(xcolor[2])),
                -1,
            )
        mask_img = mask_img.astype(np.uint8)
    else:
        mask_img[y, x] = 1
        mask_img = mask_img[:, :, None] * np.ones((1, 1, 3))
        mask_img = (mask_img * 255).astype(np.uint8)

    return mask_img


def convert_ndc_to_image(coords, img_size):
    """
    coords: (B, 2, N)
    img_size: (H, W)
    """
    coords = coords.transpose(0, 2, 1)
    coords = (coords / 2 + 0.5) * img_size[None, None, :]
    coords = coords.transpose(0, 2, 1)
    return coords
