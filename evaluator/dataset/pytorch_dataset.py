from __future__ import annotations

import copy
from typing import Union
import pytorch_lightning as L
import torch
from torch.utils.data import random_split, Subset
from nni.nas.evaluator.pytorch.lightning import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as td
import os
from .transforms import *

TRANSFORMS = {
    'cut_out': lambda length: cutout_transform(length)
}

class BaseDataset(L.LightningDataModule):
    """
    A generic PyTorch Lightning DataModule to support multiple datasets.
    """

    def __init__(self, dataset_name: str,
                 phases: list,
                 data_dir: str,
                 train_transforms=None,
                 test_transforms=None,
                 train_val_split: Union[float, list] = 0.,
                 generator_seed: int = None,
                 subset: int = None
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.phases = phases
        self.data_dir = data_dir
        self.train_val_split = train_val_split
        self.generator = torch.Generator().manual_seed(generator_seed) if generator_seed is not None else None
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.train = None
        self.val = None
        self.test = None

        self.train_bs = 32
        self.val_bs = 32
        self.test_bs = 32

        self.train_sf = True
        self.val_sf = False
        self.test_sf = False

        self.train_kwargs = {}
        self.val_kwargs = {}
        self.test_kwargs = {}

        self.input_tensor = None
        self.classes = None
        self.subset = subset
        self.prepare_data()

    def custom_transforms(self, custom_transform):
        transform_list = []
        if custom_transform is None:
            return []
        for transform in custom_transform:
            T = None
            for transform_name, params in transform.items():
                T = TRANSFORMS[transform_name](**params)
            transform_list.append(T)
        return transform_list


    def prepare_data(self):
        # To be overridden in child classes if specific behavior is needed
        pass

    def setup(self, stage: str):
        # To be overridden in child classes if specific behavior is needed
        pass

    def split_train_val(self, full_dataset):
        ld = len(full_dataset)
        if isinstance(self.train_val_split, float) or isinstance(self.train_val_split, int):
            if self.train_val_split != 0:
                if isinstance(self.train_val_split, float):
                    val_size = ld * self.train_val_split
                    self.train, self.val = random_split(full_dataset, [int(ld - val_size), int(val_size)], generator=self.generator)
                else:
                    self.train, self.val = random_split(full_dataset, [int(ld - self.train_val_split), int(self.train_val_split)], generator=self.generator)
            else:
                self.train = full_dataset
        else:
            self.train, self.val = random_split(full_dataset, self.train_val_split, generator=self.generator)



    def train_dataloader(self, **kwargs):
        if self.val is None:
            self.classes = self.train.classes
            self.input_tensor = self.train.data[0].shape
        else:
            self.classes = self.train.dataset.classes
            try:
                self.input_tensor = self.train.dataset.data[0].shape
            except:
                from PIL import Image
                img = Image.open(self.train.dataset.imgs[0][0]).convert("L")  # Convert to grayscale if needed
                self.input_tensor = img.size  # (width, height)
                if len(self.input_tensor) == 2:
                    img = np.expand_dims(np.array(img), axis=-1)  # shape becomes (H, W, 1)
                    self.input_tensor = img.shape


        if kwargs:
            self.train_bs = kwargs.pop('batch_size', 32)
            self.train_sf = kwargs.pop('shuffle', True)
            self.train_kwargs.update(kwargs)
        if self.subset is not None:
            data = Subset(self.train, range(self.subset))
        else:
            data = self.train
        return DataLoader(data, batch_size=self.train_bs, shuffle=self.train_sf, persistent_workers=False, **self.train_kwargs)

    def val_dataloader(self, **kwargs):
        if kwargs:
            if self.train_val_split != 0:
                kwargs.update(self.train_kwargs.pop('num_workers', 8))
                kwargs.update(self.train_kwargs.pop('pin_memory', True))
            self.val_bs = kwargs.pop('batch_size', 32)
            self.val_sf = kwargs.pop('shuffle', False)
            self.val_kwargs.update(kwargs)
        if self.subset is not None:
            data = Subset(self.val, range(self.subset))
        else:
            data = self.val
        if self.val is not None:
            return DataLoader(data, batch_size=self.val_bs, shuffle=self.val_sf, persistent_workers=False, **self.val_kwargs)
        else:
            return None

    def test_dataloader(self, **kwargs):
        if kwargs:
            self.test_bs = kwargs.pop('batch_size', 32)
            self.test_sf = kwargs.pop('shuffle', False)
            self.test_kwargs.update(kwargs)
        if self.subset is not None:
            data = Subset(self.test, range(self.subset))
        else:
            data = self.test
        return DataLoader(data, batch_size=self.test_bs, shuffle=self.test_sf, **self.test_kwargs)

    def __deepcopy__(self, memo):
        # Todo: find a fix for this
        # Create a shallow copy first
        copied = copy.copy(self)
        # Remove the trainer reference to prevent deepcopy issues
        copied.trainer = None
        return copied

class SweepData(BaseDataset):
    def __init__(self, phases, set: str='D1', imsize: int = 128, data_dir: str = './', train_transforms=None, test_transforms=None, **kwargs):
        train_transform_list = [
           transforms.Grayscale(),
           transforms.Resize((imsize, imsize)),
           transforms.ToTensor(),
        ] + self.custom_transforms(train_transforms)

        test_transforms_list = [
           transforms.Grayscale(),
           transforms.Resize((imsize, imsize)),
           transforms.ToTensor(),
        ] + self.custom_transforms(test_transforms)
        self.set = set

        super().__init__('SweepData', phases, data_dir, transforms.Compose(train_transform_list),
                         transforms.Compose(test_transforms_list),
                         **kwargs)

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = td.ImageFolder(f'F:/Datasets/RAiSD-AI_Dataset/{self.set}/train', transform=self.train_transforms)
            self.split_train_val(full_dataset)
        elif stage == "test":
            self.test = td.ImageFolder(f'F:/Datasets/RAiSD-AI_Dataset/{self.set}/test', transform=self.test_transforms)


# Specific dataset classes
class CIFAR10(BaseDataset):
    def __init__(self, phases, data_dir: str = './', train_transforms=None, test_transforms=None, **kwargs):
        train_transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ] + self.custom_transforms(train_transforms)

        test_transforms_list = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ] + self.custom_transforms(test_transforms)

        super().__init__('CIFAR10', phases, data_dir, transforms.Compose(train_transform_list),
                         transforms.Compose(test_transforms_list),
                         **kwargs)

    def prepare_data(self):
        td.CIFAR10(f'{self.data_dir}/train/{self.dataset_name}/', train=True, download=True)
        td.CIFAR10(f'{self.data_dir}/test/{self.dataset_name}/', train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = td.CIFAR10(f'{self.data_dir}/train/{self.dataset_name}/', train=True, transform=self.train_transforms)
            self.split_train_val(full_dataset)
        elif stage == "test":
            self.test = td.CIFAR10(f'{self.data_dir}/test/{self.dataset_name}/', train=False, transform=self.test_transforms)


class CIFAR100(BaseDataset):
    def __init__(self, phases, data_dir: str = './', train_transforms=None, test_transforms=None, **kwargs):
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ] + self.custom_transforms(train_transforms)

        test_transforms = [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ] + self.custom_transforms(test_transforms)

        super().__init__('CIFAR100', phases, data_dir, transforms.Compose(train_transforms),
                         transforms.Compose(test_transforms),
                         **kwargs)

    def prepare_data(self):
        td.CIFAR100(f'{self.data_dir}/train/{self.dataset_name}/', train=True, download=True)
        td.CIFAR100(f'{self.data_dir}/test/{self.dataset_name}/', train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = td.CIFAR100(f'{self.data_dir}/train/{self.dataset_name}/', train=True, transform=self.train_transforms)
            self.split_train_val(full_dataset)
        elif stage == "test":
            self.test = td.CIFAR100(f'{self.data_dir}/test/{self.dataset_name}/', train=False, transform=self.test_transforms)


class MNIST(BaseDataset):
    def __init__(self, phases, data_dir: str = './', train_transforms=None, test_transforms=None, **kwargs):
        train_transforms = [
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ] + self.custom_transforms(train_transforms)
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize([0.1307], [0.3081])
        ] + self.custom_transforms(test_transforms)
        super().__init__('MNIST', phases, data_dir, transforms.Compose(train_transforms),
                         transforms.Compose(test_transforms),
                         **kwargs)

    def prepare_data(self):
        td.MNIST(f'{self.data_dir}/train/{self.dataset_name}/', train=True, download=True)
        td.MNIST(f'{self.data_dir}/test/{self.dataset_name}/', train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = td.MNIST(f'{self.data_dir}/train/{self.dataset_name}/', train=True, transform=self.train_transforms)
            self.split_train_val(full_dataset)
        elif stage == "test":
            self.test = td.MNIST(f'{self.data_dir}/test/{self.dataset_name}/', train=False, transform=self.test_transforms)


class SVHN(BaseDataset):
    def __init__(self, phases, data_dir: str = './', train_transforms=None, test_transforms=None, **kwargs):
        train_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ] + self.custom_transforms(train_transforms)
        test_transforms = [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ] + self.custom_transforms(test_transforms)
        super().__init__('SVHN', phases, data_dir, transforms.Compose(train_transforms),
                         transforms.Compose(test_transforms),
                         **kwargs)

    def prepare_data(self):
        td.SVHN(f'{self.data_dir}/train/{self.dataset_name}/', split='train', download=True)
        td.SVHN(f'{self.data_dir}/val/{self.dataset_name}/', split='extra', download=True)
        td.SVHN(f'{self.data_dir}/test/{self.dataset_name}/', split='test', download=True)

    def setup(self, stage: str):
        if stage == "fit":
            self.train = td.SVHN(f'{self.data_dir}/train/{self.dataset_name}/', split='train', transform=self.train_transforms)
            self.val = td.SVHN(f'{self.data_dir}/val/{self.dataset_name}/', split='extra', transform=self.train_transforms)
        elif stage == "test":
            self.test = td.SVHN(f'{self.data_dir}/test/{self.dataset_name}/', split='test', transform=self.test_transforms)


class ImageNet(BaseDataset):
    def __init__(self, phases, data_dir: str = './', train_transforms=None, test_transforms=None, **kwargs):
        train_transforms = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ] + self.custom_transforms(train_transforms)

        test_transforms = [
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ] + self.custom_transforms(test_transforms)

        super().__init__('ImageNet', phases, data_dir, transforms.Compose(train_transforms),
                         transforms.Compose(test_transforms),
                         **kwargs)

    def prepare_data(self):
        td.ImageNet(f'{self.data_dir}/train/{self.dataset_name}/', train=True, download=True)
        td.ImageNet(f'{self.data_dir}/test/{self.dataset_name}/', train=False, download=True)

    def setup(self, stage: str):
        if stage == "fit":
            full_dataset = td.ImageNet(f'{self.data_dir}/train/{self.dataset_name}/', train=True, transform=self.train_transforms)
            self.split_train_val(full_dataset)
        elif stage == "test":
            self.test = td.ImageNet(f'{self.data_dir}/test/{self.dataset_name}/', train=False, transform=self.test_transforms)
