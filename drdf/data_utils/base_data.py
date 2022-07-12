import collections
import os
import os.path as osp
import pdb
import stat
from lib2to3.pytree import Base
from typing import Any, Dict, List, Tuple

import imageio
import numpy as np
import torch
import trimesh
from torch.utils.data.dataloader import default_collate

from ..utils import image as image_utils
from ..utils import ray_utils


class BaseData:
    def __init__(self, opts):
        self.opts = opts
        self.img_size = opts.DATALOADER.IMG_SIZE
        return

    @staticmethod
    def sample_points_mesh(mesh, npoints):
        points, faceinds = trimesh.sample.sample_surface(mesh, npoints)
        points = np.array(points)
        return points

    @staticmethod
    def sample_ray_points(mesh, n_rays):
        sampled_pts = BaseData.sample_points_mesh(mesh, n_rays)

        return

    @staticmethod
    def sample_along_rays_old(points, rays, pts_per_ray, sampling_strat, **kwargs):
        if "uniform" in sampling_strat:
            sampled_lambda = np.random.uniform(
                low=0,
                high=1,
                size=(
                    1,
                    pts_per_ray,
                    len(points),
                ),
            )
            # max_depth = kwargs['clip_z_max'] * (
            #     hitting_depth[None, None, ] * 0 + 1
            # ) * (-1)  ## depths are negative.

            sampled_lambda = sampled_lambda * max_depth

        elif "normal" in sampling_strat:
            if "variange" in kwargs.keys():
                variance = kwargs["variance"]
            else:
                variance = 0.1
            sampled_lambda = np.random.normal(
                0,
                variance,
                size=(
                    1,
                    pts_per_ray,
                    len(points),
                ),
            )

        else:
            assert False, "no sampling strategy specificed"
        rays = rays[:, None].repeat(pts_per_ray, axis=1)
        new_points = points + rays * sampled_lambda
        new_points = new_points.reshape(3, -1)
        return new_points

    @staticmethod
    def sample_along_rays(rays, pts_per_ray, sampling_strat, bounds, **kwargs):
        lambda_bounds = BaseData.compute_lambda_bounds(rays, bounds)
        ray_origin = rays[..., 0:3]
        ray_dir = rays[..., 3:]  ## N x 3
        if sampling_strat == "normal":
            if "variance" in kwargs.keys():
                variance = kwargs["variance"]
            else:
                variance = 0.4
            sampled_lambda = np.random.normal(
                0,
                variance,
                size=(
                    len(rays),
                    pts_per_ray,
                ),
            )
            sampled_lambda = np.clip(
                sampled_lambda,
                a_min=lambda_bounds[:, None, 0],
                a_max=lambda_bounds[:, None, 1],
            )
        elif sampling_strat == "uniform":
            sampled_lambda = np.random.uniform(
                low=0,
                high=1,
                size=(
                    len(rays),
                    pts_per_ray,
                ),
            )
            sampled_lambda = lambda_bounds[:, 0, None] + sampled_lambda * (
                lambda_bounds[:, 1, None] - lambda_bounds[:, 0, None]
            )  ## N x pts_per_ray
        else:
            assert False, "Unkown sampling strat"
        points = (
            ray_origin[:, None, :] + sampled_lambda[:, :, None] * ray_dir[:, None, :]
        )

        new_ray_dir = points * 0 + ray_dir[:, None, :]
        new_rays = np.concatenate([points, new_ray_dir], axis=2)
        return new_rays

    @staticmethod
    def compute_lambda_bounds(rays, bounds):

        ray_origin = rays[..., 0:3]
        ray_dir = rays[..., 3:6]
        lambda_bounds = (
            bounds[
                None,
            ]
            - ray_origin[
                :,
                None,
            ]
        )
        lambda_bounds = ray_utils.signed_divide(lambda_bounds, ray_dir[:, None])
        lambda_bounds.sort(axis=1)
        lambda_bounds = np.stack(
            [np.max(lambda_bounds[:, 0], axis=1), np.min(lambda_bounds[:, 1], axis=1)],
            axis=1,
        )
        return lambda_bounds

    @staticmethod
    def sample_rays(mesh, RT, Kndc, z_max, ray_dir_lst, npoints_lst):
        all_points = []
        for ray_dir, npoints in zip(ray_dir_lst, npoints_lst):
            points = BaseData.sample_along_ray_dir_helper(
                mesh, RT, Kndc, z_max, ray_dir, npoints
            )
            all_points.append(points)
        all_points = np.concatenate(all_points, axis=0)
        return all_points

    @staticmethod
    def sample_along_ray_dir_helper(mesh, RT, Kndc, z_max, ray_dir_type, npoints):
        points_world = BaseData.sample_points_mesh(mesh, npoints)
        view_frust_pts = ray_utils.compute_view_frustrum(RT, Kndc, z_max=z_max)

        bounds = np.vstack(
            [
                np.min(view_frust_pts, axis=1),  ## min
                np.max(view_frust_pts, axis=1),
            ]  ## max
        )

        if ray_dir_type == "C":
            ray_dir, hitting_depth = ray_utils.get_camera_ray_dir(
                points_world, RT, return_hitting_depth=True
            )

        elif ray_dir_type in ["X", "Y", "Z"]:
            ray_dir = ray_utils.get_axis_ray_dir(points_world, axis=ray_dir_type)
        else:
            assert False, "Unknown ray dir"

        rays = np.concatenate(
            [points_world, ray_dir], axis=1
        )  ## ray 0:3 origins, 3:6, directions.

        uniform_pts = BaseData.sample_along_rays(
            rays, 256, sampling_strat="uniform", bounds=bounds
        )
        normal_pts = BaseData.sample_along_rays(
            rays, 256, sampling_strat="normal", bounds=bounds
        )
        pts = np.concatenate([uniform_pts, normal_pts], axis=1)  ## [point_loc, ray_dir]
        return pts

    @staticmethod
    def sample_random(low, high, rngstate):
        high = max(high, low)
        b1 = low + (high - low) // 3
        b2 = low + ((high - low) * 2) // 3
        if not rngstate:
            rngstate = np.random

        p = rngstate.random()
        if p < 0.9:
            if p > 0.45:
                low = b2
                high = high
            else:
                low = low
                high = b1
        else:
            low = b1
            high = b2
        # print(f"low {low} high {high}")

        high = max(high, low)
        if high == low:
            return low
        else:
            return rngstate.randint(low=low, high=high)

    @staticmethod
    def get_item_data(dataset_dir, metadata, preprocess_depth, **crop_kwargs):
        house_id = metadata["house_id"]
        img_name = metadata["image_names"]

        img = BaseData.get_img(dataset_dir, house_id, img_name)
        depth = BaseData.get_depth(dataset_dir, house_id, img_name, preprocess_depth)

        cam2world, intrinsics = BaseData.forward_camera(metadata)
        mp3d_int = intrinsics * 1
        mp3d_img_size = img.shape[0:2]
        camera_rt = np.linalg.inv(cam2world)

        (
            img,
            depth,
            intrinsics,
            K_crop,
            bbox,
            resize_function,
            high_res_img,
            depth_high_res,
        ) = BaseData.center_crop_image(img, depth, intrinsics, **crop_kwargs)

        if True:
            img_size = img.shape[1], img.shape[0]
            Kndc_mult = np.array(
                [[2.0 / img_size[0], 0, -1], [0, 2.0 / img_size[1], -1], [0, 0, 1]]
            )
            kNDC = np.matmul(Kndc_mult, intrinsics)

        img_data = {}

        img_data["img"] = ((img * 1) / 255.0).astype(np.float32)
        img_data["depth"] = depth
        img_data["kNDC"] = kNDC
        img_data["intrinsics"] = intrinsics
        img_data["RT"] = camera_rt
        img_data["mp3d_int"] = mp3d_int
        img_data["mp3d_img_size"] = mp3d_img_size
        img_data["mp3d_2d_bbox"] = bbox
        img_data["depth_hr"] = np.array(depth_high_res * 1)
        return img_data

    @staticmethod
    def forward_camera(camera_dict) -> Tuple[np.array, np.array]:
        cam2world = camera_dict["cam2world"]
        intrinsic = camera_dict["intrinsics"]
        Rx = trimesh.transformations.euler_matrix(np.pi, 0, 0, "sxyz")
        cam2world = np.matmul(cam2world, Rx)  ## this handles matterport transformation.
        return cam2world, intrinsic

    @staticmethod
    def get_depth(dataset_dir, house_id, img_name, preprocess_depth, **crop_kwargs):
        depth_path = BaseData.get_depth_path(dataset_dir, house_id, img_name)
        return preprocess_depth(depth_path)

    @staticmethod
    def get_img(dataset_dir, house_id, img_name):
        img_path = BaseData.get_img_path(dataset_dir, house_id, img_name)
        img = imageio.imread(img_path)
        return img

    @staticmethod
    def get_depth_path(dataset_dir, house_id, img_name):
        img_dir = osp.join(
            dataset_dir,
            "undistorted_depth_images",
            house_id,
            "undistorted_depth_images",
        )
        img_name = img_name.replace("_i", "_d").replace(".jpg", ".png")
        depth_path = osp.join(img_dir, img_name)
        return depth_path

    @staticmethod
    def get_img_path(dataset_dir, house_id, img_name):
        img_dir = osp.join(
            dataset_dir,
            "undistorted_color_images",
            house_id,
            "undistorted_color_images",
        )
        img_path = osp.join(img_dir, img_name)
        return img_path

    @staticmethod
    def center_crop_image(
        img, depth, intrinsics, img_size, rngstate=None, scaling=0.7, split="train"
    ):
        # np.random.seed()

        min_img_size = min(img.shape[0], img.shape[1])
        half_size = (min_img_size * scaling) // 2
        half_size = int(half_size)

        if half_size > img.shape[1] - half_size - 1:
            tx = half_size
        else:
            tx = BaseData.sample_random(
                low=half_size,
                high=img.shape[1] - half_size - 1,
                rngstate=rngstate,
            )

        if half_size > img.shape[0] - half_size - 1:
            ty = half_size
        else:
            ty = BaseData.sample_random(
                low=half_size,
                high=img.shape[0] - half_size - 1,
                rngstate=rngstate,
            )

        if split == "val":
            tx, ty = int(img.shape[1] / 2), int(img.shape[0] / 2)

        center = (img.shape[1] // 2, img.shape[0] // 2)
        bbox = np.array(
            [tx - half_size, ty - half_size, tx + half_size, ty + half_size]
        )

        K_crop = np.eye(3)
        high_res_img = img = img[
            ty - half_size : ty + half_size, tx - half_size : tx + half_size
        ]
        depth_high_res = depth = depth[
            ty - half_size : ty + half_size, tx - half_size : tx + half_size
        ]

        high_res_img = np.array(high_res_img)
        transK = np.eye(3)
        transK[0, 2] = 1 * (0 - (tx - half_size))
        transK[1, 2] = 1 * (0 - (ty - half_size))
        K_crop = np.matmul(transK, K_crop)
        intrinsics = np.matmul(transK, intrinsics)
        out_img_size = img.shape[0]
        scale = img_size / out_img_size
        img, scale_factor = image_utils.resize_img(img, scale)

        depth, scale_factor = image_utils.resize_img(depth, scale)
        scaleK = np.eye(3)
        scaleK[0, 0] = scale_factor[0]
        scaleK[1, 1] = scale_factor[1]
        intrinsics = np.matmul(scaleK, intrinsics)
        K_crop = np.matmul(scaleK, transK)

        def resize_function(
            inpimg,
        ):
            inpimg = inpimg[bbox[1] : bbox[3], bbox[0] : bbox[1]]
            inpimg = image_utils.resize_img(inpimg, scale)
            return inpimg

        return (
            img,
            depth,
            intrinsics,
            K_crop,
            bbox,
            resize_function,
            high_res_img,
            depth_high_res,
        )


# -------- Collate Function --------#
# ----------------------------------#
def recursive_convert_to_torch(elem):
    if torch.is_tensor(elem):
        return elem
    elif type(elem).__module__ == "numpy":
        if elem.size == 0:
            return torch.zeros(elem.shape).type(torch.DoubleTensor)
        else:
            return torch.from_numpy(elem)
    elif isinstance(elem, int):
        return torch.LongTensor([elem])
    elif isinstance(elem, float):
        return torch.DoubleTensor([elem])
    elif isinstance(elem, collections.Mapping):
        return {key: recursive_convert_to_torch(elem[key]) for key in elem}
    elif isinstance(elem, collections.Sequence):
        return [recursive_convert_to_torch(samples) for samples in elem]
    elif elem is None:
        return elem
    else:
        return elem


def collate_fn(batch):
    """Globe data collater.
    Assumes each instance is a dict.
    Applies different collation rules for each field.
    Args:
        batch: List of loaded elements via Dataset.__getitem__
    """
    collated_batch = {"empty": True}
    new_batch = []
    for b in batch:
        if not b["empty"]:
            new_batch.append(b)

    if len(new_batch) > 0:
        for key in new_batch[0]:
            # print(key)
            if key == "mesh" or key == "obj_mesh":
                collated_batch[key] = recursive_convert_to_torch(
                    [elem[key] for elem in new_batch]
                )
            else:
                collated_batch[key] = default_collate([elem[key] for elem in new_batch])
        collated_batch["empty"] = False
    return collated_batch


def worker_init_fn(worker_id):
    ppid = os.getppid()
    np.random.seed(ppid + worker_id)
