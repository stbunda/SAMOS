"""
Operations found in MobileNet, MobileNetV2 and MobileNetV3
based on:
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

"""
from typing import Literal, Any

import torch
import torch.nn as nn
from torch import Tensor
import lightning.pytorch as pl

from convolution_modules import ConvBlock

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MB_ConvBlock(ConvBlock):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer=None):
        padding = (kernel_size - 1) // 2
        super().__init__(input_channels,
                         output_channels,
                         kernel_size,
                         stride,
                         padding)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(output_channels),
            nn.ReLU6(inplace=True)
        )

class MB_InvertedResidual(pl.LightningModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 stride: int,
                 expand_ratio: int,
                 norm_layer=None):
        super(MB_InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(input_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(MB_ConvBlock(input_channels, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            MB_ConvBlock(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, output_channels, 1, 1, 0, bias=False),
            norm_layer(output_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)