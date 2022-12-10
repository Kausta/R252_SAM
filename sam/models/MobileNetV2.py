'''Adapted version of mobile net v2 that runs on CIFAR 10'''

import torch 
import torch.nn as nn
from sam.util import ConfigType

class MobileNetV2(nn.Module):
    def __init__(self, config: ConfigType): # class_num = 10 for CIFAR-10
        super(MobileNetV2, self).__init__()
        self.class_num = config.model.num_outputs
        self.pretrained = config.model.pretrained

        # load model #####################################
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'mobilenet_v2',
                                    pretrained=self.pretrained)

        # set last layer ################################
        if self.model.classifier[1].out_features != self.class_num:
            self.model.classifier[1] = nn.Linear(1280, self.class_num)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out

