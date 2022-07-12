import copy
import os.path as osp
import pdb
import random

import imageio
import numpy as np
import ray
import torch
import torch.nn as nn
from loguru import logger
from torch.profiler import ProfilerActivity, profiler, record_function

from ..data_utils import matterport_rgbd as matterport3d_data
from ..html_utils import visdom_visualizer as visdom_utils
from ..nnutils import vol_dist
from ..renderer import render_utils
from ..utils import intersection_finder_utils, utils_3d
from . import base_trainer, scene_visuals
from .scene_visuals import SceneVisuals


class SceneRGBDTrainer(base_trainer.BaseTrainer, SceneVisuals):
    def __init__(self, opts):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.enable_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(opts)

    def init_model(
        self,
    ):
        opts = self.opts
        model = vol_dist.VolSDF(opts)
        self.model = model.to(self.device)

        self.visdom_logger = visdom_utils.Visualizer(opts)
        self.render_func = eval(opts.RENDERER.RENDER_FUNC)

        if opts.TRAIN.NUM_PRETRAIN_EPOCHS > 0:
            self.load_network(
                self.model,
                network_label="model",
                epoch_label=f"epoch_{opts.TRAIN.NUM_PRETRAIN_EPOCHS}",
            )
        self.intersection_finder = intersection_finder_utils.intersection_finder_drdf

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
            # self.val_dataloader = self.dataloader
            self.val_batch = False
            self.prev_loss = 0.0
        else:
            assert False, "dataset not available"
        return

    def init_optimizer(
        self,
    ):
        opts = self.opts
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opts.OPTIM.LEARNING_RATE,
            betas=(opts.OPTIM.BETA1, opts.OPTIM.BETA2),
        )
        len_dataloader = len(self.dataloader)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=opts.OPTIM.LEARNING_RATE,
            epochs=opts.TRAIN.NUM_EPOCHS,
            steps_per_epoch=len_dataloader,
            cycle_momentum=False,
            pct_start=0.005,
        )
        return

    def set_input(self, batch):
        if batch["empty"] == True:
            self.invalid_batch = True
            return

        opts = self.opts
        input_imgs = batch["image"]
        batch_size = input_imgs.size(0)
        self.input_imgs = input_imgs.to(self.device)
        self.depth_gt = batch["depth"]
        self.RT = batch["RT"].to(self.device)
        points = batch["points"].reshape(batch_size, -1, 6)
        points = points.permute(0, 2, 1).to(self.device)

        self.points = points[:, :3, :]
        self.points = self.points.requires_grad_(True)
        self.ray_dirs = points[:, :3, :]
        self.Kndc = batch["kNDC"].to(self.device)
        self.index = batch["index"]

        if True:
            self.meshes = batch["mesh"]
            self.mp3d_int = batch["mp3d_int"].numpy()
            self.mp3d_img_size = batch["mp3d_img_size"].numpy()
            self.mp3d_2d_bbox = batch["mp3d_2d_bbox"].numpy()

        validity = batch["distance_validity"].reshape(batch_size, -1)

        num_rays = batch["penalty_types"].shape[1]
        grad_penalty_mask_lst = []
        for bx in range(batch_size):
            for rx in range(num_rays):
                last_ind = torch.where(batch["penalty_types"][bx][rx] > 0)[0]
                grad_penalty_mask = batch["penalty_types"][bx][rx] * 0
                if len(last_ind) > 0:
                    last_ind = last_ind[-1]
                    grad_penalty_mask[:last_ind] = 1
                grad_penalty_mask_lst.append(grad_penalty_mask)

        grad_penalty_mask = torch.stack(grad_penalty_mask_lst).reshape(batch_size, -1)
        self.grad_penalty_mask = grad_penalty_mask.to(self.device)
        penalty_types = batch["penalty_types"].reshape(batch_size, -1)
        penalty_types = penalty_types * (validity == 1) - 1 * (validity == 0)
        self.distance = (
            batch["penalty_regions"].reshape(batch_size, -1, 3).to(self.device)
        )
        self.penalty_types = penalty_types.to(self.device)

    def forward(
        self,
    ):
        predictions, losses = self.model.forward(
            images=self.input_imgs,
            points=self.points,
            ray_dir=self.ray_dirs * 0,
            kNDC=self.Kndc,
            RT=self.RT,
            penalty_regions=self.distance,
            penalty_types=self.penalty_types,
            grad_penalty_masks=self.grad_penalty_mask,
        )

        self.loss_factors = {}
        self.total_loss = 0
        self.losses = losses
        weight_dict = {}
        # weight_dict['segments'] = 1.0
        weight_dict["io"] = 1.0
        weight_dict["oo"] = 1.0
        weight_dict["oi"] = 1.0
        weight_dict["oi_start"] = 1.0
        weight_dict["ii"] = 1.0

        # weight_dict['equality'] = 1.0
        # weight_dict['inequality'] = 1.0
        weight_dict["grad"] = 0.1

        self.weight_dict = weight_dict
        for key in losses.keys():
            if key not in weight_dict.keys():
                self.loss_factors[key] = losses[key].mean()
            else:
                self.loss_factors[key] = weight_dict[key] * losses[key].mean()
                self.total_loss += self.loss_factors[key]

        if not (self.total_loss.item() == self.total_loss.item()):
            pdb.set_trace()
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            predictions["xyz_ndc"] = self.model.module.xyz_ndc
        else:
            predictions["xyz_ndc"] = self.model.xyz_ndc

        self.tracked_scalars = predictions["tracked_scalars"]
        smooth_alpha = 0.0
        if not self.val_batch:
            for k in self.smoothed_factor_losses.keys():
                if k in self.loss_factors.keys():
                    self.smoothed_factor_losses[k] = (
                        smooth_alpha * self.smoothed_factor_losses[k]
                        + (1 - smooth_alpha) * self.loss_factors[k].item()
                    )

            self.smoothed_total_loss = (
                self.smoothed_total_loss * smooth_alpha
                + (1 - smooth_alpha) * self.total_loss.item()
            )

            if False:
                if (
                    self.prev_loss + 0.3
                ) < self.smoothed_total_loss and self.total_steps > 100:
                    breakpoint()
            self.prev_loss = self.smoothed_total_loss
            # logger.debug('prev loss {}'.format(self.prev_loss))
        self.predictions = predictions
        return

    def debug_log(
        self,
    ):
        if self.val_batch:
            return

        if True:
            self.tensorboard_writer.log_gradients_norms(self.model, self.total_steps)

        if True:
            self.tensorboard_writer.log_model_grad_norm(self.model, self.total_steps)
        return

    def define_criterion(self):
        opts = self.opts
        self.smoothed_factor_losses = {
            "segments": 0.0,
            "io": 0.0,
            "oi": 0.0,
            "ii": 0.0,
            "oo": 0.0,
            "oi_start": 0.0,
            # 'equality': 0.0,
            # 'inequality': 0.0,
            "grad": 0.0,
        }
        return

    def val(self, num_batches=5):
        opts = self.opts
        self.val_batch = True
        val_losses = []
        device = self.device
        self.model.eval()
        for i, batch in enumerate(self.val_dataloader):
            self.set_input(batch)
            self.forward()
            val_loss = self.total_loss.item()
            val_losses.append(val_loss)
            if i >= num_batches and num_batches > 0:
                break
            # self.total_loss.backward(torch.FloatTensor(0).device)
            # self.optimizer.zero_grad()
        self.model.train()
        self.val_batch = False
        val_loss = np.mean(np.array(val_losses))

        return val_loss

    @staticmethod
    def compute_grad_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def backward(
        self,
    ):
        opts = self.opts
        self.optimizer.zero_grad()
        self.total_loss.backward()

        self.grad_unclipped = self.compute_grad_norm(self.model)
        if opts.OPTIM.GRAD_CLIPPING.ENABLED:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), opts.OPTIM.GRAD_CLIPPING.MAX_NORM_VALUE
            )
        self.grad = self.compute_grad_norm(self.model)
        self.optimizer.step()
        self.scheduler.step()
        return

    def get_current_scalars(
        self,
    ):
        ineqpts = (
            ((self.penalty_types < 5) * (self.penalty_types > 1)).float().mean().item()
        )
        eqpts = (
            torch.logical_or(self.penalty_types == 5, self.penalty_types == 1)
            .float()
            .mean()
            .item()
        )
        lr = self.scheduler.get_last_lr()[0]
        time_per_batch = self.time_per_batch / self.epoch_iter
        loss_dict = {
            "total_loss": self.smoothed_total_loss,
            "iter_frac": self.real_iter * 1.0 / self.total_steps,
            "valid_points": self.predictions["valid_points"].float().mean().item(),
            "lr": lr,
            "val_loss": self.val_loss,
            "ineq_pts": ineqpts,
            "eq_pts": eqpts,
            "time_per_batch": time_per_batch,
        }

        for k in self.smoothed_factor_losses.keys():
            loss_dict["loss_" + k] = self.smoothed_factor_losses[k]
        return loss_dict

    def log_step(self, total_steps, epoch, epoch_iter):
        opts = self.opts
        scalars = self.get_current_scalars()
        tracked_scalars = self.tracked_scalars
        self.visdom_logger.print_current_scalars(epoch, epoch_iter, scalars)

        if opts.LOGGING.PLOT_SCALARS:
            for key, value in scalars.items():
                self.tensorboard_writer.add_scalar(f"train/{key}", value, total_steps)
            self.tensorboard_writer.add_scalar(
                "train/grad_unclipped", self.grad_unclipped, self.total_steps
            )
            self.tensorboard_writer.add_scalar(
                "train/grad", self.grad, self.total_steps
            )
            for key, value in tracked_scalars.items():
                self.tensorboard_writer.add_scalar(f"tracked/{key}", value, total_steps)
        if False:
            self.debug_log()
            # if opts.LOGGING.PLOT_SCALARS:
            #     self.visdom_logger.plot_current_scalars(
            #         epoch,
            #         float(epoch_iter) / self.dataset_size, opts, scalars
            #     )
        return

    def visuals_to_save(self, visual_count):
        batch_visuals = []
        opts = self.opts
        img_size = np.array([opts.DATALOADER.IMG_SIZE, opts.DATALOADER.IMG_SIZE])
        predictions = self.predictions
        project_points = predictions["project_points"].data.cpu().numpy()
        # project_points_depth = predictions['xyz_ndc'].data.cpu().numpy()[:,2,:]
        project_points_depth = predictions["points_cam"].data.cpu().numpy()[:, 2, :]
        # project_points = utils_3d.convert_ndc_to_image(project_points, img_size)
        valid_points = predictions["valid_points"]
        for bx in range(visual_count):
            visuals = {}
            visuals["image"] = (
                self.input_imgs[bx].detach().cpu().numpy().transpose(1, 2, 0)
            )
            visuals["points_proj"] = utils_3d.render_mesh_cv2(
                project_points[bx], project_points_depth[bx], img_size, color_depth=True
            )
            if False:
                visuals["pred_mesh"] = self.generate_mesh(bx)

            if True:
                visuals["depth_pred"] = SceneVisuals.colored_depthmap(
                    self.generate_depth(bx), d_min=0.0, d_max=10.0
                )

            if True:
                visuals["depth_gt"] = SceneVisuals.colored_depthmap(
                    self.depth_gt[bx], d_min=0.0, d_max=10.0
                )
                # pdb.set_trace()
            # png_path = osp.join(opts.RENDER_DIR, 'pred_sdf_mesh.png')
            # rendered_visuals = SceneVisuals.render_scene(
            #     mesh_path=visuals['pred_mesh'], png_path=png_path, bIndex=bx
            # )
            # visuals['pred_img'] = rendered_visuals['image']
            # visuals['pred_depth'] = rendered_visuals['depth']
            # visuals['pred_normal'] = rendered_visuals['normal']
            batch_visuals.append(visuals)
        return batch_visuals

    def get_visuals(self, visual_count):
        visuals = self.visuals_to_save(visual_count)
        return visuals

    def log_visuals(self, total_steps):
        opts = self.opts
        visuals = self.get_visuals(visual_count=opts.LOGGING.VISUAL_COUNT)
        # self.visdom_logger.display_current_results(visuals[0], total_steps)
        prefix = "train"
        for key in visuals[0].keys():
            self.tensorboard_writer.add_image(
                f"{prefix}/{key}", visuals[0][key], total_steps, dataformats="HWC"
            )
        return


def trace_handler(p):
    breakpoint()
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")
