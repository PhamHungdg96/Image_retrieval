from __future__ import print_function
import os
import yaml
cfg=yaml.safe_load(open('config.yaml'))
type_ex='resnet'
model= cfg[type_ex]['model']
output_type=cfg[type_ex]['o_t']
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.utils.model_zoo as model_zoo
from PIL import Image
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
dimension_out=2048

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
        x = self.layer4(x)  # x after layer4, shape = N * 2048 * H/32 * W/32
        max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
        Max = max_pool(x)  # avg.size = N * 2048 * 1 * 1
        Max = Max.view(Max.size(0), -1)  # avg.size = N * 2048
        avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 2048 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 2048
        output = {
            'max': Max,
            'avg': avg
        }
        return output

class Restnet_Ex(object):
    def __init__(self,model=model,output_type=output_type,normalize=False):
        self.dimension=dimension_out
        self.net=ResidualNet(model=model,pretrained=True)
        self.transforms=transforms.Compose([transforms.Resize(150),
                                            transforms.RandomCrop(150),
                                            transforms.ToTensor()
                                        ])
        self.normalize  = normalize
        self.output_type=output_type
    def feature(self, input):
        assert isinstance(input, str)
        img=Image.open(input).convert('RGB')
        x=torch.unsqueeze(self.transforms(img),dim=0)
        y=self.net(x)['avg'].data[0].numpy()
        if self.normalize:
            y /= np.sum(y)
        return y


if __name__ == "__main__":
    restnet_Ex = Restnet_Ex()
    print(restnet_Ex.dimension)
    input = '../../data/style/0_0_001.png'
    hist = restnet_Ex.feature(input)
    input = '../../data/style/0_0_006.png'
    hist2 = restnet_Ex.feature(input)
    print(np.sum((hist - hist2) ** 2))