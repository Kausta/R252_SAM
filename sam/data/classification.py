import os
from pathlib import Path
import torch
import torch.utils.data as data

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

            if config.model.model_cls == "MobileNetV3" and config.model.mobile_net_pretrained:
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
        else:
            raise ValueError("Unknown dataset")
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]

        return img, target
