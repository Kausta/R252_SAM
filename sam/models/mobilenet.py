import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights

from sam.util import ConfigType

__all__ = ["WideResnet"]

class MobileNetV3(nn.Module):
    def __init__(self, config: ConfigType):
        super().__init__()

        pretrained = config.model.mobile_net_pretrained
        if config.model.mobile_net_small:
            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        
        num_outputs = config.model.num_outputs
        last_layer = self.model.classifier[-1]
        if last_layer.out_features != num_outputs:
            self.model.classifier[-1] = nn.Linear(last_layer.in_features, num_outputs)
            nn.init.normal_(self.model.classifier[-1].weight, 0, 0.01)
            nn.init.zeros_(self.model.classifier[-1].bias)

    def forward(self, x):
        return self.model(x)