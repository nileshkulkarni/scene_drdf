import numpy as np
import torch
import torch.nn as nn
import torchvision


def turnNormOff(
    model,
):
    turnBNoff(model)
    turnGNoff(model)


def turnBNoff(
    model,
):
    for m in model.modules():
        if (
            isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
        ):
            m.eval()
        if isinstance(m, nn.SyncBatchNorm):
            m.eval()


def turnGNoff(
    model,
):
    for m in model.modules():
        if (
            isinstance(m, nn.BatchNorm1d)
            or isinstance(m, nn.BatchNorm2d)
            or isinstance(m, nn.BatchNorm3d)
        ):
            m.eval()
