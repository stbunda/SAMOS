""" Operations """
try:
    from typing import Literal, Any
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl


class ConvBlock(pl.LightningModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 1,
                 affine: bool = True):
        super().__init__()

        padding = kernel_size // 2 if padding is None else padding

        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class ResBlock(pl.LightningModule):
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 affine: bool = True):
        super().__init__()

        padding = kernel_size // 2 if padding is None else padding
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine),
        )

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(output_channels, output_channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine),
        )

        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False))
        elif input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(output_channels, affine=affine),
            )
        else:
            self.downsample = None


    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = self.conv2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return torch.add(x, y)

class DilBlock(pl.LightningModule):
    """ (Dilated) depthwise separable conv
        ReLU - (Dilated) depthwise separable - Pointwise - BN

        If dilation == 2, 3x3 conv => 5x5 receptive field
                          5x5 conv => 9x9 receptive field
        """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 2,
                 affine: bool = True):
        super().__init__()
        # padding = kernel_size // 2 if padding is None else padding
        padding = 2 if kernel_size == 3 else 4  # Calculate padding size based on kernel size

        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, dilation=dilation,
                      groups=input_channels, bias=False),
            nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SepBlock(pl.LightningModule):
    """
    Depthwise separable conv
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 affine: bool = True):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size, stride=stride, padding=padding, groups=input_channels,
                      bias=False),
            nn.Conv2d(input_channels, output_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DilSepConv(pl.LightningModule):
    """
        Depthwise separable conv using Dilution
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 dilation: int = 2,
                 affine: bool = True):
        super().__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.net = nn.Sequential(
            DilBlock(input_channels, input_channels, kernel_size, stride, padding, dilation=dilation, affine=affine),
            DilBlock(input_channels, output_channels, 1, 1, 0, dilation=dilation, affine=affine)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Conv1x7_7x1(pl.LightningModule):
    """
    A PyTorch module that performs a sequence of convolutions:
    1. A 1x7 convolution followed by
    2. A 7x1 convolution.
    """

    def __init__(self,
                 input_channels: int,
                 stride: int = 1,
                 affine: bool = True):
        """
        Initializes the Conv1x7Then7x1 module.

        Args:
            in_size (int): The number of input channels.
            stride (int): The stride for the convolutions.
            affine (bool): Whether to use affine parameters in batch normalization. Default is True.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, input_channels, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
            nn.Conv2d(input_channels, input_channels, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(input_channels, affine=affine),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SEBlock(pl.LightningModule):
    """
    A PyTorch module implementing a Squeeze-and-Excitation (SE) block.

    This block is designed to recalibrate channel-wise feature responses by explicitly modeling interdependencies between channels.

    Attributes:
        squeeze (nn.AdaptiveAvgPool2d): The adaptive average pooling layer to squeeze the spatial dimensions.
        excitation (nn.Sequential): The sequential container holding the fully connected layers and activation functions for the excitation operation.
    """

    def __init__(self,
                 input_channels: int,
                 reduction: int = 4):
        """
        Initializes the SE_Block module.

        Args:
            input_channels (int): The number of input channels.
            reduction (int): The reduction ratio for the excitation operation. Default is 4.
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_channels // reduction, input_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the squeeze and excitation operations.
        """
        batch_size, channels, _, _ = x.shape
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return torch.multiply(x, y)


class MBConvBlock(pl.LightningModule):
    """
    A PyTorch module implementing a Mobile Inverted Bottleneck Convolution (MBConv) block.

    This block is designed to efficiently capture spatial features with depthwise separable convolutions and includes a Squeeze-and-Excitation (SE) block for channel-wise attention.

    Attributes:
        relu (nn.ReLU): The ReLU activation function.
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, ReLU activation, and SE block.
        conv2 (nn.Sequential): The sequential container for the shortcut connection.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 expansion: int = 1,
                 affine: bool = True):
        super().__init__()

        if output_channels >= 64:
            expansion = 4

        expanded_channels = output_channels * expansion
        padding = kernel_size // 2 if padding is None else padding

        self.relu = nn.ReLU(inplace=False)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, expanded_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expanded_channels, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride=1, padding=padding,
                      groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels, affine=affine),

            nn.ReLU(inplace=False),
            SEBlock(expanded_channels),

            nn.ReLU(inplace=False),
            nn.Conv2d(expanded_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

        self.shortcut = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, output_channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
                Defines the forward pass of the module.

                Args:
                    inputs1 (torch.Tensor): The primary input tensor.
                    inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

                Returns:
                    torch.Tensor: The output tensor after applying the MBConv operations and adding the shortcut connection.
                """
        y1 = self.op(x)
        y2 = self.shortcut(x)
        out = torch.add(y1, y2)
        return out


class FusedMBConvBlock(pl.LightningModule):
    """
    A PyTorch module implementing a Fused Mobile Inverted Bottleneck Convolution (Fused MBConv) block.

    This block is designed to efficiently capture spatial features with depthwise separable convolutions and includes a Squeeze-and-Excitation (SE) block for channel-wise attention.

    Attributes:
        relu (nn.ReLU): The ReLU activation function.
        op (nn.Sequential): The sequential container holding the convolutional operations, batch normalization, ReLU activation, and SE block.
        conv2 (nn.Sequential): The sequential container for the shortcut connection.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 expansion: int = 1,
                 affine: bool = True):
        super().__init__()

        if output_channels >= 64:
            expansion = 4

        expanded_channels = output_channels * expansion
        padding = kernel_size // 2 if padding is None else padding

        self.relu = nn.ReLU(inplace=False)
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, expanded_channels, kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(expanded_channels, affine=affine),

            nn.ReLU(inplace=False),
            SEBlock(expanded_channels),

            nn.ReLU(inplace=False),
            nn.Conv2d(expanded_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

        self.shortcut = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, output_channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
                Defines the forward pass of the module.

                Args:
                    inputs1 (torch.Tensor): The primary input tensor.
                    inputs2 (torch.Tensor): The secondary input tensor for the shortcut connection.

                Returns:
                    torch.Tensor: The output tensor after applying the MBConv operations and adding the shortcut connection.
                """
        y1 = self.op(x)
        y2 = self.shortcut(x)
        out = torch.add(y1, y2)
        return out


class Bottleneck(pl.LightningModule):
    """
    A PyTorch module implementing a Linear Bottleneck block.

    This block is designed to reduce the number of parameters and computational cost while maintaining performance.
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = None,
                 affine: bool = True):
        super().__init__()

        reduction = 4 if output_channels >= 128 else 2
        padding = kernel_size // 2 if padding is None else padding
        temp_out = output_channels // reduction

        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, temp_out, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(temp_out, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(temp_out, temp_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(temp_out, affine=affine),

            nn.ReLU(inplace=False),
            nn.Conv2d(temp_out, output_channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(input_channels, output_channels, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_channels, affine=affine)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the module.

        Args:
            inputs1 (torch.Tensor): The primary input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the MBConv operations and adding the shortcut connection.
        """
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        out = torch.add(y1, y2)
        return out


class AuxiliaryHead(pl.LightningModule):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 img_size: int = 32):
        super().__init__()
        stride = 2
        if img_size == 32:
            stride = 3
        else:
            # assuming input size 14x14
            stride = 2

        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            nn.Conv2d(input_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
