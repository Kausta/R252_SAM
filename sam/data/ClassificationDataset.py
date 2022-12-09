import os
from pathlib import Path
import torch
import torch.utils.data as data

from torchvision import transforms as T
import torchvision.datasets as datasets

import sam.nn as SNN

from sam.util.config import ConfigType

__all__ = ["ClassificationDataset"]

class ClassificationDataset(data.Dataset):
    def __init__(self, phase: str, config: ConfigType):
        super().__init__()
        assert phase in ["train", "val", "test"]

        self.data_root = str(Path(config.data.data_root).expanduser() / config.data.dataset)
        self.phase = phase

        dataset = config.data.dataset
        assert dataset in ["cifar10"]
        
        training = self.phase == "train"
        if dataset == "cifar10":
            transforms = []
            if training:
                if config.data.use_autoaugment:
                    transforms.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
                transforms.extend([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(32, padding=4, padding_mode="reflect")
                ])
            transforms.extend([
                T.ToTensor(),
                # From original jax sam implementation 
                T.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
            ])
            if training and config.data.use_cutout:
                transforms.append(SNN.Cutout(length=16, inplace=True))
            self.dataset = datasets.CIFAR10(self.data_root, train=training, transform=T.Compose(transforms), download=False)
        else:
            raise ValueError("Unknown dataset")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        return img, target
