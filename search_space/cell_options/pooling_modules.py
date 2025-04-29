""" Operations """
import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl

class MaxPool(pl.LightningModule):
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2,
                 padding: int = 1):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x)

class AvgPool(pl.LightningModule):
    def __init__(self,
                 kernel_size: int = 2,
                 stride: int = 2,
                 padding: int = 1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x)

class GAvgPool(pl.LightningModule):
    def __init__(self,
                 output_size: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)