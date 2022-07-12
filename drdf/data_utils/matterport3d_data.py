import collections
import io
import itertools
import os
import os.path as osp
import pdb
import pickle as pkl

# from ..utils import grid_utils
import zipfile
from typing import Any, DefaultDict, Dict, List, Tuple

import cv2
import imageio
import numpy as np
import torch
import trimesh
from fvcore.common.config import CfgNode
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from ..utils import geometry_utils, matterport_parse, mesh_utils, ray_utils
from ..utils.default_imports import *
from . import base_data
from .base_data import BaseData, collate_fn, worker_init_fn

curr_dir = osp.dirname(osp.abspath(__file__))

cachedir = osp.join(osp.dirname(osp.abspath(__file__)), "../..", "cachedir")


class MatterportData(BaseData):
    def __init__(self, opts: CfgNode):
        super().__init__(opts)
        self.dataset_dir = opts.MATTERPORT_PATH

        self.filter_keys = [
            "_i1_",  # "_i2_", "_i0_"
        ]  # we are using only these as the i0 and i2 point to the bottom
        _house_ids = os.listdir(osp.join(self.dataset_dir))
        _house_ids = [f for f in _house_ids if f != "sens"]
        split_dir = osp.join(cachedir, "splits")
        split_file = osp.join(split_dir, "mp3d_split.pkl")
        # # house_ids = os.listdir(
        # #     osp.join(self.dataset_dir, 'undistorted_camera_parameters')
        # # )
        with open(split_file, "rb") as f:
            splits = pkl.load(f)
        # splits = read_house_splits(split_file, house_ids)

        _house_ids = splits[opts.DATALOADER.SPLIT]
        self.meshes = DefaultDict(dict)
        self.mp3d_cachedir = osp.join(cachedir, "mp3d_data_zmx8")
        # _house_ids = _house_ids[10:11]
        # _house_ids = ["r1Q1Z4BcV1o"]
        _house_ids = ["17DRP5sb8fy"]

        # breakpoint()
        filter_by_region = True
        _indexed_items = matterport_parse.preload_conf_files(
            matterport_dir=self.dataset_dir,
            house_ids=_house_ids,
        )

        ## current hardcoded to get only one item. You can remove the indexing below to get more items.
        _indexed_items = matterport_parse.filter_indices(
            _indexed_items, self.filter_keys, keep_valid_regions=filter_by_region
        )

        _indexed_items = [
            _indexed_items[i] for i in self.mark_good_inds(_indexed_items, "train")
        ]
        # _indexed_items = [_indexed_items[i] for i in self.mark_good_inds(_indexed_items, opts.DATALOADER.SPLIT)]
        self.houses2mesh = matterport_parse.preload_meshes(
            indexed_items=_indexed_items, matterport_dir=self.dataset_dir
        )

        # self.houses2mesh = matterport_parse.preload_meshes(
        #     indexed_items=list(itertools.chain(*_indexed_items)), matterport_dir=self.dataset_dir
        # )
        # self.houses2mesh = matterport_parse.preload_meshes(
        #     indexed_items=list(_indexed_items), matterport_dir=self.dataset_dir
        # )

        # trimesh.exchange.export.export_mesh(
        #     trimesh.Trimesh(
        #         vertices=self.houses2mesh['JmbYfDe2QKZ']['vertices'],
        #         faces=self.houses2mesh['JmbYfDe2QKZ']['faces']
        #     ),
        #     'house_og.ply',
        # )

        self.indexed_items = _indexed_items
        self.num_items = len(self.indexed_items)
        print("num_items", self.num_items)
        return

    def check_example_validity(self, meta_data):
        img_data = self.forward_img(meta_data)
        depth = img_data["depth"]
        if self.opts.DATALOADER.FILTER_DEPTH:
            depth1d = depth.reshape(-1)
            if (depth1d < 3.0).mean() > 0.6:
                elem = {"empty": True}
                return elem
            if (depth1d < 0.001).mean() > 0.5:
                elem = {"empty": True}
                # print('end {} '.format(index))
                return elem

        # region_mesh = self.forward_mesh(meta_data)
        # points = base_data.BaseData.sample_points_mesh(
        #     region_mesh, npoints=10000
        # )
        # extrinsics = np.linalg.inv(img_data['cam2mp'])
        # points = points.transpose() * 1
        # points = geometry_utils.transform_points(points, extrinsics)

        # if np.mean(points[2] < 0.2) > 0.7:
        #     elem = {'empty': True}
        #     imageio.imsave('test.png', img_data['img'])
        #     pdb.set_trace()
        #     return elem
        return {"empty": False}

    def mark_good_inds(self, indexed_items, split):
        opts = self.opts
        checkpoint_dir = osp.join(cachedir, "matterport_good_list", "all")
        # checkpoint_dir = osp.join(cachedir, 'snapshots', opts.NAME)
        if split == "val":
            good_inds_file = osp.join(checkpoint_dir, f"good_inds_file_{split}.pkl")
        else:
            good_inds_file = osp.join(checkpoint_dir, "good_inds_file.pkl")
        logger.info(f"Good Inds file {good_inds_file}")
        if osp.exists(good_inds_file):
            logger.info("Fast loading good inds from file")
            with open(good_inds_file, "rb") as f:
                good_list = pkl.load(f)
        else:
            good_list = []
            for ix in range(len(indexed_items)):
                meta_data = indexed_items[ix]
                elem = self.check_example_validity(meta_data)
                if ix % 1000 == 0:
                    logger.info(f"Checking {ix}/{len(indexed_items)}")
                if not elem["empty"]:
                    uuid = meta_data["uuid"]
                    good_list.append(uuid)

            with open(good_inds_file, "wb") as f:
                pkl.dump(good_list, f)

        good_inds = []

        good_list = set(good_list)
        for ix in range(len(indexed_items)):
            meta_data = indexed_items[ix]
            uuid = meta_data["uuid"]
            if uuid in good_list:
                good_inds.append(ix)

        return good_inds

    def load_mesh(self, meta_data):
        house_id = meta_data["house_id"]

        mesh = self.houses2mesh[house_id]
        mesh = {
            "faces": mesh["faces"] * 1,
            "vertices": mesh["vertices"] * 1,
        }
        (
            fids,
            face_labels,
            face_segments,
            fids_c,
            fids_w,
        ) = matterport_parse.load_face_metadata(
            metadata=meta_data, face_cachedir=self.mp3d_cachedir
        )

        if fids is not None:
            # mesh['face_labels'] =
            mesh["faces"] = mesh["faces"][fids]
            mesh = matterport_parse.delete_unused_verts(
                mesh,
            )
            mesh = trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"])
            # print('time to load {}'.format(time.time()- sttime))
            return mesh
        else:
            return None

    def forward_mesh(self, meta_data):
        house_id = meta_data["house_id"]
        region_id = meta_data["region_id"]
        region_mesh_pth = osp.join(
            self.dataset_dir, house_id, "region_segmentations", f"region{region_id}.ply"
        )
        mesh = None  ## caching meshes!
        if house_id in self.meshes.keys():
            if region_id in self.meshes["house_id"].keys():
                mesh = self.meshes["house_id"]["region_id"]
        if mesh is None:
            with open(region_mesh_pth, "rb") as f:
                mesh = trimesh.exchange.ply.load_ply(f)
            self.meshes[house_id][region_id] = mesh

        simple_mesh = trimesh.Trimesh(mesh["vertices"], mesh["faces"])
        return simple_mesh

    def forward_img(self, meta_data) -> Dict[str, Any]:
        opts = self.opts
        cam2mp, intrinsics = self.forward_camera(meta_data)
        mp3d_int = intrinsics * 1

        img_name = meta_data["image_names"]
        house_id = meta_data["house_id"]
        img_dir = osp.join(
            self.dataset_dir,
            "undistorted_color_images",
            house_id,
            "undistorted_color_images",
        )
        depth_dir = osp.join(
            self.dataset_dir,
            "undistorted_depth_images",
            house_id,
            "undistorted_depth_images",
        )
        # depth_dir = '/scratch/justincj_root/justincj/nileshk/SceneTSDF/Matterport3d/raw/{}/undistorted_depth_images'.format(house_id)
        depth_path = osp.join(
            depth_dir, img_name.replace("_i", "_d").replace(".jpg", ".png")
        )

        img_path = osp.join(img_dir, img_name)
        # print(img_path)
        img = imageio.imread(img_path)
        depth = imageio.imread(depth_path)
        og_img_size = (img.shape[0], img.shape[1])
        bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        # mg, depth, intrinsics, K_crop, bbox, resize_function, high_res_img
        (
            img,
            depth,
            intrinsics,
            K_crop,
            bbox,
            resize_function,
            high_res_img,
            depth_high_res,
        ) = self.center_crop_image(
            img,
            depth,
            intrinsics,
            img_size=self.img_size,
            scaling=1.0,
            split=opts.DATALOADER.SPLIT,
        )
        # depth = depth.astype(float) * 0.25
        # depth = depth / 1000
        img = img.astype(np.float32) / 255
        img = img.transpose(2, 0, 1)
        data = {}
        data["img"] = img
        data["depth"] = depth
        data["cam2mp"] = cam2mp
        data["intrinsics"] = intrinsics
        data["mp3d_int"] = mp3d_int
        data["mp3d_img_size"] = og_img_size
        data["mp3d_2d_bbox"] = bbox
        return data

    def forward_img_seq(self, meta_lst) -> Dict:
        data_seq = []
        for ix in range(len(meta_lst)):
            data_seq.append(self.forward_img(meta_lst[ix]))
        return data_seq

    def project_points_ndc(self, points, img_size, K, img=None):
        if True:  # Filter points.2
            valid_inds = np.where(points[2, :] > -6)[0]
            points = points[:, valid_inds]

        projected_points = geometry_utils.perspective_transform(points, K)
        # img_points = np.matmul(K, points).transpose(1, 0)
        # # pdb.set_trace()
        # img_points[:, 0:2] = img_points[:, 0:2] / img_points[:, 2:3]

        x = projected_points[0, :] * img_size[0] / 2 + img_size[0] / 2
        y = projected_points[1, :] * img_size[1] / 2 + img_size[1] / 2

        x = np.round(np.array(x)).astype(int)
        y = np.round(np.array(y)).astype(int)

        valid = np.logical_and(
            np.logical_and(x >= 0, x < img_size[0]),
            np.logical_and(y >= 0, y < img_size[1]),
        )
        valid_inds = np.where(valid)[0]
        # project to image plane
        x = x[valid_inds]
        y = y[valid_inds]

        mask_img = np.zeros((img_size[1], img_size[0], 3))
        mask_img[y, x, :] = 255

        if img is not None:
            mask_img = np.concatenate([img, mask_img], axis=1)
        imageio.imsave("mask.png", mask_img)
        return mask_img

    def __len__(
        self,
    ):
        return self.num_items

    def __getitem__(self, index):
        opts = self.opts

        index = 25
        ref_meta_data = self.indexed_items[index]
        # meta_data_lst = self.indexed_items[0]
        # ref_meta_data = meta_data_lst[0] ## current set to the first view
        mesh = self.load_mesh(
            ref_meta_data
        )  ## this gives the mesh of the complete house

        img_data = self.forward_img(
            ref_meta_data
        )  ## contains img, depth, cam_ext, cam_int
        image = img_data["img"]
        extrinsics = np.linalg.inv(img_data["cam2mp"])

        img_size = (image.shape[2], image.shape[1])
        elem = {"empty": True}
        K = img_data["intrinsics"] * 1

        # sp_views_data = self.forward_img_seq(meta_data_lst[1:])
        if True:
            K[0, 0] = 1 * K[0, 0]  # Flip the x-axis.
            K[1, 1] = 1 * K[1, 1]  # Flip the y-axis.
            K[2, 2] = 1  # Flip the z-axis.

            Kndc_mult = np.array(
                [[2.0 / img_size[0], 0, -1], [0, 2.0 / img_size[1], -1], [0, 0, 1]]
            )

            Kndc = np.matmul(Kndc_mult, K)

        def is_point_inside(points_temp):
            RT = extrinsics
            points_temp = geometry_utils.transform_points(points_temp.transpose(), RT)
            xyz_ndc = geometry_utils.perspective_transform(points_temp, Kndc)
            xyz_valids = (xyz_ndc >= -1) * (xyz_ndc <= 1)

            xyz_valids = xyz_valids[0, :] * xyz_valids[1, :]
            return xyz_valids

        if False:  ## this is sample code to verfiy camera poses.
            points = self.sample_points_mesh(mesh, 100000)

            points = points.transpose() * 1
            points = geometry_utils.transform_points(points, extrinsics)
            proj_img = self.project_points_ndc(points, img_size, Kndc, img=image)
            imageio.imsave("test.png", proj_img)
            pdb.set_trace()

        if mesh is None:
            elem["empty"] = True
            return elem
        ## Ray based sampling startegy!
        try:
            points = self.sample_rays(
                mesh,
                RT=extrinsics,
                Kndc=Kndc,
                z_max=opts.DATALOADER.SAMPLING.Z_MAX,
                ray_dir_lst=opts.DATALOADER.SAMPLING.RAY_DIR_LST,
                npoints_lst=opts.DATALOADER.SAMPLING.N_RAYS_LST,
            )
        except Exception as e:
            logger.error(f"error {e}")
            logger.error(f"Errror while dataloading the index {index}")
            logger.error(f"ignoring the index {index}")

        # mesh_utils.save_mesh(mesh, "test.ply")
        # mesh_utils.save_point_cloud(points[0, :, :3], "pcl.ply")
        # breakpoint()

        points_shape = points.shape  ## N, N_per_ray, 6
        points = points.reshape(-1, 6)  ## first 3 are point, next 3 are direc
        ray_dist, int_loc, valid_int, tri_ids = ray_utils.get_special_ray_distances(
            mesh,
            points=points[:, 0:3],
            rays=points[:, 3:],
            unsigned_ray_dist=opts.DATALOADER.UNSIGNED_RAY_DIST,
            signed=opts.DATALOADER.SIGNED_RAY_DIST,
        )
        elem["points"] = points.transpose(1, 0).astype(
            np.float32
        )  ## array of points and directions.
        elem["ray_dist"] = ray_dist.astype(np.float32)
        elem["mesh"] = mesh
        elem["image"] = image
        elem["RT"] = extrinsics.astype(np.float32)
        elem["kNDC"] = Kndc.astype(np.float32)
        elem["extents"] = np.array(mesh.extents)
        elem["valid_intersect"] = valid_int.astype(np.float32)
        elem["img_size"] = img_size
        elem["mp3d_int"] = img_data["mp3d_int"]
        elem["mp3d_img_size"] = np.array(img_data["mp3d_img_size"])
        elem["mp3d_2d_bbox"] = img_data["mp3d_2d_bbox"]
        elem["index"] = index
        elem["empty"] = False
        # elem['img_data'] = img_data
        return elem


def matterport_dataloader(opts, shuffle=False):
    dataset = MatterportData(opts)
    shuffle = opts.DATALOADER.SPLIT == "train" or shuffle
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=opts.TRAIN.NUM_WORKERS,
        collate_fn=base_data.collate_fn,
        worker_init_fn=base_data.worker_init_fn,
    )
    return dataloader


if __name__ == "__main__":
    from ..config import defaults
    from ..utils import parse_args

    cmd_args = parse_args.parse_args()
    cfg = defaults.get_cfg_defaults()
    if cmd_args.cfg_file is not None:
        cfg.merge_from_file(cmd_args.cfg_file)
    if cmd_args.set_cfgs is not None:
        cfg.merge_from_list(cmd_args.set_cfgs)

    dataset = MatterportData(cfg)
    dp = dataset[0]
    import pdb

    pdb.set_trace()
