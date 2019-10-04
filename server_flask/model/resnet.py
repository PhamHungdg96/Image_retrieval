from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo

import numpy as np 
import os

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
class ResidualNet(ResNet):
    def __init__(self, model='resnet152', pretrained=True):
        if model == "resnet18":
            super().__init__(BasicBlock, [2, 2, 2, 2], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        elif model == "resnet34":
            super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        elif model == "resnet50":
            super().__init__(Bottleneck, [3, 4, 6, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        elif model == "resnet101":
            super().__init__(Bottleneck, [3, 4, 23, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        elif model == "resnet152":
            super().__init__(Bottleneck, [3, 8, 36, 3], 1000)
            if pretrained:
                self.load_state_dict(model_zoo.load_url(model_urls['resnet152']))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
        max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
        Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
        Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
        avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        fc = self.fc(avg)  # fc.size = N * 1000
        output = {
            'max': Max,
            'avg': avg,
            'fc' : fc
        }
        return output
