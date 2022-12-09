from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

import torchvision.transforms.functional as TF

__all__ = ["Cutout"]

class Cutout(nn.Module):
    def __init__(self, length: int = 16, inplace:bool = False):
        super().__init__()

        self.length = length
        self.inplace = inplace

    @staticmethod
    def get_params(img: Tensor, length: int) -> Tuple[int, int, int, int]:
        """Get parameters for ``erase`` for a cutout. 
        Based on jax sam implementation and torchvision RandomErasing

        Args:
            img (Tensor): Tensor image to apply cutout to.
            length (int): Cutout length.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        _, img_h, img_w = TF.get_dimensions(img)
        
        center_height = torch.randint(0, img_h, size=(1,)).item()
        center_width = torch.randint(0, img_w, size=(1,)).item()

        i = max(0, center_height - length // 2) # lower_pad
        i2 = max(0, img_h - center_height - length // 2) # upper_pad
        j = max(0, center_width - length // 2) # left_pad
        j2 = max(0, img_w - center_width - length // 2) # right_pad

        h = img_h - (i + i2)
        w = img_w - (j + j2)

        v = 0.0

        return i, j, h, w, v

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x, y, h, w, v = self.get_params(img, length=self.length)
        return TF.erase(img, x, y, h, w, v, self.inplace)
