from .convolution_modules import ConvBlock, ResBlock, AuxiliaryHead, SepBlock, DilBlock, Bottleneck, SEBlock, \
    Conv1x7_7x1, MBConvBlock, FusedMBConvBlock
# from .pooling_modules import MaxPool, AvgPool, GAvgPool
# from .operation_modules import Identity, Sum, Concatenate, Zero

# from torchvision.ops.misc import Conv2dNormActivation as ConvBlock
from torchvision.models.mobilenetv2 import InvertedResidual as MB2_InvertedResidual

from .operation_modules import Identity, Sum, Concatenate, Zero
from .pooling_modules import MaxPool, AvgPool, GAvgPool

__all__ = [
    # Static operations
    'Identity', 'Sum', 'Concatenate', 'Zero',
    # Pooling operations
    'MaxPool', 'AvgPool', 'GAvgPool',
    # Conv operations MobileNet
    'MB2_InvertedResidual',
    # CGP operations todo: fix
    'ConvBlock', 'ResBlock', 'AuxiliaryHead', 'SepBlock', 'DilBlock', 'Bottleneck', 'SEBlock', 'Conv1x7_7x1',
    'MBConvBlock', 'FusedMBConvBlock'
]
