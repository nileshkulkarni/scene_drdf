import os.path as osp
import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from . import dist_3d_utils
from . import image as image_utils


class SceneVisuals:
    def __init__(self, opts):
        self.opts = opts
        return

    @staticmethod
    def resize_img(
        opts,
        img,
        bbox,
    ):
        img = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        img_size = opts.DATALOADER.IMG_SIZE
        scale = img_size / img.shape[0]
        try:
            img, _ = image_utils.resize_img(img, scale)
        except Exception as e:
            img = np.zeros((img_size, img_size, 3))
        return img

    def render_mesh(self, mesh_path, png_path, bIndex):
        opts = self.opts
        visuals = {}
        bbox = self.mp3d_2d_bbox[bIndex]
        rend_img, rend_depth, rend_normal = self.render_func(
            mesh_path,
            png_path,
            np.linalg.inv(self.RT[bIndex].cpu().numpy()),
            intrinsic=self.mp3d_int[bIndex],
            img_width=self.mp3d_img_size[bIndex][1],
            img_height=self.mp3d_img_size[bIndex][0],
            max_depth=opts.MODEL.ZMAX + 1,
        )
        rend_depth = np.stack([rend_depth, rend_depth, rend_depth], axis=2)
        rend_depth = rend_depth / (opts.MODEL.ZMAX + 1) * 255
        rend_depth = rend_depth.astype(np.uint8)
        try:
            rend_img = SceneVisuals.resize_img(opts, rend_img, bbox)
            rend_depth = SceneVisuals.resize_img(opts, rend_depth, bbox)
            rend_normal = SceneVisuals.resize_img(opts, rend_normal, bbox)
        except Exception as e:
            rend_img = np.zeros((opts.IMG_SIZE, opts.IMG_SIZE, 3)).astype(np.uint8)
            rend_depth = rend_img
            rend_normal = rend_img

        visuals["img"] = (rend_img * 255).astype(np.uint8)
        visuals["depth"] = rend_depth
        visuals["normal"] = rend_normal
        return visuals

    def generate_mesh(self, bIndex):
        opts = self.opts
        ind = self.index[bIndex].item()
        meta_data = {}
        meta_data["img"] = self.input_imgs[bIndex]
        meta_data["RT"] = self.RT[bIndex]
        meta_data["Kndc"] = self.Kndc[bIndex]

        meta_data["z_max"] = opts.DATALOADER.SAMPLING.Z_MAX
        save_dir = self.html_vis.get_save_dir(self.total_steps)
        save_path = osp.join(save_dir, f"{ind}_pred_mesh.ply")
        meta_data["save_path"] = save_path
        is_training = self.model.training
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # net = copy.deepcopy(self.model.module)
            net = self.model.module
        else:
            # net = copy.deepcopy(self.model)
            net = self.model

        net.eval()
        try:
            mesh = dist_3d_utils.gen_mesh_from_net(
                opts=opts,
                net=net,
                device=f"cuda:{0}",
                meta_data=meta_data,
                intersection_finder=self.intersection_finder,
            )
            if mesh is not None:
                trimesh.exchange.export.export_mesh(mesh, save_path)
            mesh_path = save_path

            if is_training:
                net.train()
            return mesh, mesh_path
        except ValueError as e:
            mesh_path = ""
            if is_training:
                self.model.train()
            return None, mesh_path

    def generate_depth(self, bIndex):
        opts = self.opts
        ind = self.index[bIndex].item()
        meta_data = {}
        meta_data["img"] = self.input_imgs[bIndex]
        meta_data["RT"] = self.RT[bIndex]
        meta_data["Kndc"] = self.Kndc[bIndex]
        meta_data["z_max"] = opts.DATALOADER.SAMPLING.Z_MAX
        save_dir = self.html_vis.get_save_dir(self.total_steps)
        save_path = osp.join(save_dir, f"{ind}_pred_mesh.obj")
        meta_data["save_path"] = save_path
        is_training = self.model.training
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            # net = copy.deepcopy(self.model.module)
            net = self.model.module
        else:
            # net = copy.deepcopy(self.model)
            net = self.model

        net.eval()
        try:
            depth_img = dist_3d_utils.gen_depth_from_net(
                opts=opts,
                net=net,
                device=f"cuda:{0}",
                meta_data=meta_data,
                intersection_finder=self.intersection_finder,
                resolution=256,
            )
            if is_training:
                net.train()
            return depth_img
        except ValueError as e:
            depth_img = np.zeros((opts.IMG_SIze, opts.IMG_SIZE))
            if is_training:
                self.model.train()
            return depth_img

    @staticmethod
    def colored_depthmap(depth, d_min=None, d_max=None):
        if d_min is None:
            d_min = np.min(depth)
        if d_max is None:
            d_max = np.max(depth)
        depth_relative = (depth - d_min) / (d_max - d_min)
        depth_colored = 255 * plt.cm.plasma(depth_relative)[:, :, :3]  # H, W, C
        depth_colored = depth_colored.astype(np.uint8)
        return depth_colored
