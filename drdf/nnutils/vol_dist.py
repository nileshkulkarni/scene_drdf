import math
import pdb

import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models.resnet as resnet
from torchvision import transforms

from ..nnutils import net_blocks as nb
from ..utils import geometry_utils
from . import resnetfc


class ResNet(nn.Module):
    def __init__(self, model="resnet18", nlayers=4):
        super().__init__()
        if model == "resnet18":
            net = resnet.resnet18(pretrained=True)
        elif model == "resnet34":
            net = resnet.resnet34(pretrained=True)
        elif model == "resnet50":
            net = resnet.resnet50(pretrained=True)
        else:
            raise NameError("Unknown Fan Filter setting!")

        self.nlayers = nlayers
        nb.turnBNoff(net)

        self.conv1 = net.conv1
        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool

    def forward(self, image):
        """
        :param image: [BxC_inxHxW] tensor of input image
        :return: list of [BxC_outxHxW] tensors of output features
        """
        y = image
        feat_pyramid = []
        y = self.layer0(y)
        feat_pyramid.append(y)
        if self.nlayers > 0:
            y = self.layer1(self.pool(y))
            feat_pyramid.append(y)
        if self.nlayers > 1:
            y = self.layer2(y)
            feat_pyramid.append(y)
        if self.nlayers > 2:
            y = self.layer3(y)
            feat_pyramid.append(y)
        if self.nlayers > 3:
            y = self.layer4(y)
            feat_pyramid.append(y)

        gfeat = self.avgpool.forward(y)
        gfeat = gfeat.squeeze(3).squeeze(2)
        return feat_pyramid, gfeat


def build_backbone(model="resnet18", nlayers=4):
    resnet = ResNet(model, nlayers=nlayers)
    if model == "resnet18":
        return resnet, 1024
    elif model == "resnet34":
        return resnet, 1024
    else:
        assert False, "Model backone not available"


def encoder_3d(nlayers, nc_input, nc_l1, nc_max):
    modules = []
    nc_output = nc_l1
    nc_step = 1
    for nl in range(nlayers):
        if (nl >= 1) and (nl % nc_step == 0) and (nc_output <= nc_max * 2):
            nc_output *= 2

        modules.append(nb.conv3d(False, nc_input, nc_output, stride=1))
        nc_input = nc_output
        # modules.append(nb.conv3d(False, nc_input, nc_output, stride=1))
        # modules.append(torch.nn.MaxPool3d(kernel_size=2, stride=2))

    encoder = nn.Sequential(*modules)
    return encoder, nc_output


class SpatialEncoder(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.image_encoder, feat_dim = build_backbone(model="resnet34")
        self.latent_scaling = torch.FloatTensor([0, 0])
        self.d_out = feat_dim

    def forward(self, image):
        latents, _ = self.image_encoder.forward(image)
        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",
                align_corners=True,
            )
        self.latent = torch.cat(latents, dim=1)
        return self.latent


class VolSDF(nn.Module):
    def __init__(self, opts):
        super().__init__()
        if opts.MODEL.DECODER == "pixNerf":
            self.image_encoder = SpatialEncoder(opts)
        else:
            raise NotImplementedError("Unknown decoder!")
        self.opts = opts

        self.point_emb = resnetfc.PositionalEncoding()
        self.dir_emb = resnetfc.PositionalEncoding()
        self.surface_classifier = resnetfc.ResnetFC(
            d_in=self.point_emb.d_out + self.point_emb.d_out + self.image_encoder.d_out,
            d_out=1,
            last_op=None,
        )
        self.projection = geometry_utils.perspective_transform
        self.img_size = (opts.DATALOADER.IMG_SIZE, opts.DATALOADER.IMG_SIZE)
        self.loss_fun = torch.nn.functional.l1_loss

        self.init_transforms()
        return

    def init_transforms(
        self,
    ):
        base_transform = torch.nn.Sequential(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )
        self.base_transform = torch.jit.script(base_transform)
        return

    def transform(self, images):
        images = self.base_transform(images)
        return images

    def filter_images(self, images):
        _, _, image_height, image_width = images.shape
        images = self.transform(images)
        self.feats = self.image_encoder(images)
        return self.feats

    def convert_points_to_ndc(self, points, RT, kNDC):
        points = geometry_utils.transform_points(points, RT)
        xyz_ndc = geometry_utils.convert_to_ndc(points, kNDC, self.projection)
        return points, xyz_ndc

    def get_depth_points(self, points, ray_dir, RT, kNDC):
        points, xyz_ndc = self.convert_points_to_ndc(points, RT, kNDC)
        depth = points[:, 2:3, :]
        return depth, xyz_ndc

    def query(self, defaults=None, points=None, ray_dir=None, kNDC=None, RT=None):
        opts = self.opts
        # project 3d points to image plane
        image_width, image_height = self.img_size[0], self.img_size[1]
        assert kNDC is not None, "please supply a  calibndc"

        if RT is None:  ## transform using extrinsics
            assert False, "where is the RT matrix"

        self.points_cam, self.xyz_ndc = self.convert_points_to_ndc(points, RT, kNDC)

        xyz_valids = (self.xyz_ndc >= -1) * (self.xyz_ndc <= 1)
        xyz_valids = xyz_valids[:, 0, :] * xyz_valids[:, 1, :]

        self.project_points = (
            self.xyz_ndc[
                :,
                0:2,
            ]
            * 0
        )
        self.project_points[:, 0, :] = (
            (
                self.xyz_ndc[
                    :,
                    0,
                ]
                + 1
            )
            * image_width
            / 2
        )
        self.project_points[:, 1, :] = (
            (
                self.xyz_ndc[
                    :,
                    1,
                ]
                + 1
            )
            * image_height
            / 2
        )

        self.valid_points = xyz_valids

        feat_points = F.grid_sample(
            self.feats,
            self.xyz_ndc.permute(0, 2, 1)[:, :, None, 0:2],
            align_corners=True,
            mode="bilinear",
        )
        feat_points = feat_points.squeeze(3)
        B, feat_z, npoints = feat_points.shape
        feat_points = feat_points.permute(0, 2, 1).contiguous()
        feat_points = feat_points.reshape(B * npoints, feat_z)

        xyz_feat = self.xyz_ndc.permute(0, 2, 1).contiguous()
        xyz_feat = xyz_feat.reshape(-1, 3)
        xyz_feat = self.point_emb(xyz_feat)
        ray_dir = ray_dir.permute(0, 2, 1).contiguous()
        ray_dir_feat = self.dir_emb(ray_dir.reshape(B * npoints, 3))

        if not opts.MODEL.DIR_ENCODING:
            ray_dir_feat = ray_dir_feat * 0  ## zero out the direction encoding

        if not opts.MODEL.USE_POINT_FEATURES:
            xyz_feat = xyz_feat * 0  ## zero out the point encoding

        point_concat_feat = torch.cat([feat_points, xyz_feat, ray_dir_feat], dim=1)

        self.surface_prediction = self.surface_classifier.forward(point_concat_feat)
        self.surface_prediction = self.surface_prediction.reshape(
            B, npoints, 1
        )  ## B x npoints x 1
        self.surface_prediction = self.surface_prediction.permute(
            0, 2, 1
        )  ## B x 1 x npoints
        return self.surface_prediction

    def get_preds(self, gt=False):
        opts = self.opts
        if gt:
            raise Exception("Not implemented")
        return self.surface_prediction

    def loss(self, gt_targets, pred_targets, validity=None):
        opts = self.opts
        if opts.MODEL.APPLY_LOG_TRANSFORM:
            pred_distance = geometry_utils.apply_log_transform(
                pred_targets[:, 0, None, :]
            )
            gt_distance = geometry_utils.apply_log_transform(gt_targets)
        else:
            pred_distance = pred_targets[:, 0, None, :]
            gt_distance = gt_targets
        loss = self.loss_fun(pred_distance, gt_distance.float(), reduce=False)

        loss = torch.nan_to_num(loss)
        loss = loss.squeeze(1) * validity
        loss = {"surface": loss.mean(1)}
        return loss

    def get_error(self):
        opts = self.opts
        surfacePred = self.surface_prediction
        valid_points = self.valid_points
        gt_dist = self.distance.unsqueeze(1)
        loss = self.loss(gt_dist, surfacePred, validity=valid_points)
        return loss

    def predict(
        self,
        images,
    ):
        self.filter_images(
            images,
        )
        return

    def forward(
        self,
        images,
        points,
        ray_dir,
        kNDC,
        RT,
        distance,
    ):
        opts = self.opts
        self.filter_images(images)
        self.distance = distance

        self.surface_prediction = self.query(
            points=points, ray_dir=ray_dir, kNDC=kNDC, RT=RT
        )
        predictions = {}
        predictions["surface"] = self.surface_prediction
        predictions["valid_points"] = self.valid_points
        predictions["project_points"] = self.project_points
        predictions["xyz_ndc"] = self.xyz_ndc
        predictions["points_cam"] = self.points_cam
        # predictions['depth'] = self.depth
        # predictions = self.surface_prediction
        losses = {}
        error = self.get_error()

        if type(error) is dict:
            losses.update(error)
        else:
            losses["surface"] = error
        return predictions, losses
