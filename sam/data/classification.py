import os
from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np
import random

from torchvision import transforms as T
import torchvision.datasets as datasets

from torchvision.models import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

import sam.nn as snn

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
            if config.model.model_cls == "MobileNetV3":
                if config.model.mobile_net_small:
                    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
                else:
                    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
                t = weights.transforms()
                mean, std = t.mean, t.std
            else:
                mean, std = (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)
                
            transforms.extend([
                T.ToTensor(),
                # From original jax sam implementation 
                T.Normalize(mean, std),
            ])
            if training and config.data.use_cutout:
                transforms.append(snn.Cutout(length=16, inplace=True))
            self.dataset = datasets.CIFAR10(self.data_root, train=training, transform=T.Compose(transforms), download=True)

            if len(config.data.classes) < 10 or config.data.samples_per_class < 5000:
                data = self.dataset.data
                targets = self.dataset.targets
                classes = config.data.classes
                indicess = [[i for i, target in enumerate(targets) if target == c] for c in classes]
                for i in range(len(indicess)):
                    random.shuffle(indicess[i])
                    indicess[i] = indicess[i][:config.data.samples_per_class]
                indices = list(np.array(indicess).reshape((1, 1500))[0])
                random.shuffle(indices)
                self.dataset.data = data[indices]
                self.dataset.targets = list(np.array(targets)[indices])
                self.dataset.classes = classes
                self.dataset.class_to_idx = {k: v for k, v in self.dataset.class_to_idx.items() if v in classes}
                print('Hello')
        else:
            raise ValueError("Unknown dataset")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        return img, target
