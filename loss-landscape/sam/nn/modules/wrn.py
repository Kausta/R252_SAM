import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

__all__ = ["IdentityZeroPadSkip", "WideResnetBlock", "WideResnetGroup", "WideResnet"]

class IdentityZeroPadSkip(nn.Module):
    def __init__(self, in_channels, out_channels, stride, inplace=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.channels_to_add = self.out_channels - self.in_channels

        self.inplace = inplace

    def forward(self, out: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            inp = F.avg_pool2d(inp, self.stride)
        
        if self.channels_to_add > 0:
            B, _, H, W = inp.shape
            padding = inp.new_zeros((B, self.channels_to_add, H, W))
            inp = torch.cat((inp, padding), dim=1)
        
        if not self.inplace:
            return out + inp
        out += inp
        return out

        
class WideResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, act_before_res: bool = False):
        super().__init__()

        self.act_before_res = act_before_res

        self.pre_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.post_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )

        self.skip = IdentityZeroPadSkip(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.act_before_res:
            x = self.pre_block(x)
        skip = x
        out = x
        if not self.act_before_res:
            out = self.pre_block(out)
        out = self.post_block(out)
        return self.skip(out, skip)

class WideResnetGroup(nn.Module):
    def __init__(self, blocks_per_group: int, in_channels: int, out_channels: int, stride: int = 1, act_before_res: bool = False, use_additional_skips: bool = False):
        super().__init__()

        self.use_additional_skips = use_additional_skips

        blocks = []
        for i in range(blocks_per_group):
            blocks.append(WideResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                stride=stride if i == 0 else 1,
                act_before_res=act_before_res and not i
            ))
        self.blocks = nn.Sequential(*blocks)

        if self.use_additional_skips:
            self.skip = IdentityZeroPadSkip(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.blocks(x)
        if self.use_additional_skips:
            out = self.skip(out, x)
        return out

class WideResnet(nn.Module):
    def __init__(self, blocks_per_group: int, channel_multiplier: int, num_inputs: int, num_outputs: int, use_additional_skips: bool = False):
        super().__init__()

        self.use_additional_skips = use_additional_skips

        self.groups = nn.Sequential(
            nn.Conv2d(num_inputs, 16, kernel_size=3, stride=1, padding=1, bias=False),
            WideResnetGroup(blocks_per_group, 16, 16 * channel_multiplier, act_before_res=True, use_additional_skips=use_additional_skips),
            WideResnetGroup(blocks_per_group, 16 * channel_multiplier, 32 * channel_multiplier, stride=2, use_additional_skips=use_additional_skips),
            WideResnetGroup(blocks_per_group, 32 * channel_multiplier, 64 * channel_multiplier, stride=2, use_additional_skips=use_additional_skips)
        )
        if self.use_additional_skips:
            self.skip = IdentityZeroPadSkip(3, 64 * channel_multiplier, stride=4)
        self.head = nn.Sequential(
            nn.BatchNorm2d(64 * channel_multiplier),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64 * channel_multiplier, num_outputs)
        )

        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming normal initialization for convolutional kernels (mode = fan_out, gain=2.0)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # final dense layer use a uniform distribution U[-scale, scale] where scale = 1 / sqrt(num_classes) as per the autoaugment implementation
                num_outputs = m.weight.shape[0]
                scale = 1. / np.sqrt(num_outputs)
                nn.init.uniform_(m.weight, -scale, scale)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.groups(x)
        if self.use_additional_skips:
            out = self.skip(out, x)
        pred = self.head(out)
        return pred