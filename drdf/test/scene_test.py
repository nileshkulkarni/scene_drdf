import copy
import os.path as osp
import pdb

import imageio
import ipdb
import numpy as np
import ray
import torch
import torch.nn as nn

from ..data_utils import matterport3d_data
from ..html_utils import base_html as base_html_utils
from ..html_utils import scene_html
from ..html_utils import visdom_visualizer as visdom_utils
from ..nnutils import vol_dist
from ..renderer import render_utils
from ..train import base_trainer, scene_visuals
from ..utils import intersection_finder_utils, mesh_utils, utils_3d


class SceneTest(base_trainer.BaseTrainer, scene_visuals.SceneVisuals):
    def __init__(self, opts):
        super().__init__(opts)

    def init_model(
        self,
    ):
        opts = self.opts
        model = vol_dist.VolSDF(opts)
        self.model = model.to(self.device)

        self.visdom_logger = visdom_utils.Visualizer(opts)
        self.render_func = eval(opts.RENDERER.RENDER_FUNC)

        if opts.TRAIN.NUM_EPOCHS > 0:
            self.load_network(
                self.model,
                network_label="model",
                epoch_label=f"epoch_{opts.TRAIN.NUM_EPOCHS}",
            )
        self.intersection_finder = intersection_finder_utils.intersection_finder_drdf
        return

    def initialize(self):
        opts = self.opts
        self.init_dataset()  ## define self.dataloader
        self.init_model()  ## define self.model
        return

    def init_dataset(
        self,
    ):
        opts = self.opts
        if opts.DATALOADER.DATASET_TYPE == "matterport":
            self.dataloader = matterport3d_data.matterport_dataloader(opts)
            opts_val = copy.deepcopy(opts)
            opts_val.DATALOADER.SPLIT = "val"
            self.val_dataloader = matterport3d_data.matterport_dataloader(
                opts_val, shuffle=True
            )
        else:
            assert False, "dataset not available"
        return

    def set_input(self, batch):

        if batch["empty"] == True:
            self.invalid_batch = True
            return

        opts = self.opts
        input_imgs = batch["image"]
        batch_size = input_imgs.size(0)
        self.input_imgs = input_imgs.to(self.device)
        self.RT = batch["RT"].to(self.device)
        self.points = batch["points"][:, 0:3, :].to(self.device)
        self.ray_dirs = batch["points"][:, 3:, :].to(self.device)
        self.Kndc = batch["kNDC"].to(self.device)
        self.valid_intersect = batch["valid_intersect"].to(self.device)
        self.index = batch["index"]
        self.extents = batch["extents"]

        if True:
            self.meshes = batch["mesh"]
            self.mp3d_int = batch["mp3d_int"].numpy()
            self.mp3d_img_size = batch["mp3d_img_size"].numpy()
            self.mp3d_2d_bbox = batch["mp3d_2d_bbox"].numpy()

        distance = batch["ray_dist"].to(self.device)
        if opts.DATALOADER.SIGNED_RAY_DIST:
            distance = torch.clamp(
                distance,
                min=-1 * opts.DATALOADER.CLAMP_MAX_DIST,
                max=opts.DATALOADER.CLAMP_MAX_DIST,
            )
        else:
            distance = torch.clamp(distance, min=0, max=opts.DATALOADER.CLAMP_MAX_DIST)
        self.distance = distance
        return

    def visualize_scene(
        self,
    ):
        bx = 0
        mesh = self.generate_mesh(bx)
        depth = self.generate_depth(bx)
        breakpoint()
        mesh_file = "mesh.ply"
        mesh_utils.save_mesh(mesh, mesh_file)
        imageio.imsave("test.png", depth)

    def test(
        self,
    ):
        opts = self.opts
        self.html_vis = html_vis = scene_html.HTMLWriter(opts)
        data_iterator = iter(self.dataloader)

        for i in range(len(self.dataloader)):
            batch = next(data_iterator)
            self.set_input(batch)
            # logger.debug(f"Time per batch: {self.time_per_batch}")
            # for i, batch in enumerate(self.dataloader):
            self.total_steps = i
            self.invalid_batch = False
            self.visualize_scene()

        return
