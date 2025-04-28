# from .class_dataset import ClassDataset
# from .pair_dataset import PairDataset
# from .item_dataset import ItemDataset
# from .ijb_dataset import IJBDataset
from .pytorch_dataset import MNIST, ImageNet, CIFAR10, CIFAR100, SVHN
#from . import *

__all__ = [
    'MNIST', 'ImageNet', 'CIFAR10', 'CIFAR100', 'SVHN',
]
