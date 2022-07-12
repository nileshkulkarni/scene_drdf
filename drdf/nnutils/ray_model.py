from os import stat
from turtle import distance

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import gradient


class RayModel(nn.Module):
    def __init__(self, num_ind):
        super().__init__()
        init_tensor = torch.normal(mean=0, std=torch.zeros(num_ind) + 1) * 0
        self.model_params = torch.nn.parameter.Parameter(init_tensor)
        return

    def zeros(self):
        self.model_params.data *= 0

    def forward(self, points, ray_distance, distance_loss_type):
        predictions = self.model_params * 1
        losses = self.forward_loss(ray_distance, distance_loss_type, predictions)
        grad_loss = self.gradient_loss(predictions, points)
        losses["grad"] = grad_loss
        return losses

    def forward_loss(self, gt_targets, target_types, pred_targets, validity=None):
        ii_loss = self.loss_II(pred_targets, gt_targets)
        oo_loss = self.loss_OO(pred_targets=pred_targets, gt_targets=gt_targets)
        io_loss = self.loss_IO(pred_targets=pred_targets, gt_targets=gt_targets)
        oi_loss = self.loss_OI(pred_targets=pred_targets, gt_targets=gt_targets)
        oi_loss_start = self.loss_OI_start(
            pred_targets=pred_targets, gt_targets=gt_targets
        )

        loss = (
            ii_loss * (target_types == 1)
            + oo_loss * (target_types == 2)
            + io_loss * (target_types == 3)
            + oi_loss * (target_types == 4)
            + oi_loss_start * (target_types == 5)
        )
        validity = target_types > 0
        loss = loss[validity == True]
        losses = {}
        losses["equality"] = loss.mean()
        return losses

    def gradient_loss(self, pred_targets, points):
        y_diff = pred_targets[1:] - pred_targets[:-1]
        x_diff = points[1:] - points[:-1]
        x_diff = torch.norm(x_diff, dim=1)
        grad = y_diff / (x_diff)
        loss_grad = (grad.data > 0) + (grad.data <= 0) * torch.abs(grad - (-1))
        # loss_grad = loss_grad * (target_types > 0)
        loss_grad = loss_grad.mean()
        return loss_grad

    @staticmethod
    def loss_II(pred_targets, gt_targets):
        above = gt_targets[:, 1]
        below = gt_targets[:, 0]
        sCloser = torch.abs(below) < torch.abs(above)
        i1_loss = F.l1_loss(pred_targets, below, reduce=False)
        i2_loss = F.l1_loss(pred_targets, above, reduce=False)
        ii_loss = i1_loss * (sCloser == True) + i2_loss * (sCloser == False)
        return ii_loss

    @staticmethod
    def loss_OO(pred_targets, gt_targets):
        above = gt_targets[:, 1]
        below = gt_targets[:, 0]
        mid = (below + above) / 2  ## ((s -z) + (e - z) )/2
        halfWidth = (above - below) / 2  ##  ((s-z) - (e-z))/2
        loss_oo = F.relu(halfWidth - torch.abs(pred_targets - mid))
        return loss_oo

    @staticmethod
    def loss_IO(pred_targets, gt_targets):
        above = gt_targets[:, 1]
        below = gt_targets[:, 0]
        sCloser = torch.abs(below) < torch.abs(above)
        i1_loss = F.l1_loss(pred_targets, below, reduce=False)
        o2_loss = torch.min(
            F.relu(above - pred_targets), F.l1_loss(pred_targets, below, reduce=False)
        )
        loss_io = i1_loss * (sCloser == True) + o2_loss * (sCloser == False)
        return loss_io

    @staticmethod
    def loss_OI(pred_targets, gt_targets):
        above = gt_targets[:, 1]
        below = gt_targets[:, 0]
        sCloser = torch.abs(below) < torch.abs(above)
        i2_loss = F.l1_loss(pred_targets, above, reduce=False)
        o1_loss = torch.min(
            F.relu(pred_targets - below), F.l1_loss(pred_targets, above, reduce=False)
        )
        loss_oi = o1_loss * (sCloser == True) + i2_loss * (sCloser == False)
        return loss_oi

    @staticmethod
    def loss_OI_start(pred_targets, gt_targets):
        above = gt_targets[:, 1]
        i2_loss = F.l1_loss(pred_targets, above, reduce=False)
        return i2_loss
