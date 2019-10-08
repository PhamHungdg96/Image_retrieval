from __future__ import print_function
import os
import yaml
cfg=yaml.safe_load(open('config.yaml'))
type_ex='vgg'
model= cfg[type_ex]['model']
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision.models.vgg import VGG
import torch.utils.model_zoo as model_zoo
from PIL import Image

dimension_out=512
cfg_vgg = {
  'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
  'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
  'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
ranges = {
  'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
  'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
  'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
  'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return nn.Sequential(*layers)
os.path.join

class VGGNet(VGG):
  def __init__(self, pretrained=True, model='vgg19', requires_grad=False, remove_fc=False, show_params=False):
    super().__init__(make_layers(cfg_vgg[model]))
    self.ranges = ranges[model]

    if pretrained:
      exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

    if not requires_grad:
      for param in super().parameters():
        param.requires_grad = False

    if remove_fc:  # delete redundant fully-connected layer params, can save memory
      del self.classifier

    if show_params:
      for name, param in self.named_parameters():
        print(name, param.size())

  def forward(self, x):
    x = self.features(x)

    avg_pool = torch.nn.AvgPool2d((x.size(-2), x.size(-1)), stride=(x.size(-2), x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
    avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
    avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
    return avg
class Vggnet_Ex(object):
    def __init__(self,model=model,normalize=False):
        self.dimension=dimension_out
        self.net=VGGNet(model=model,pretrained=True)
        self.transforms=transforms.Compose([transforms.Resize(150),
                                            transforms.RandomCrop(150),
                                            transforms.ToTensor()
                                        ])
        self.normalize  = normalize
    def feature(self, input):
        assert isinstance(input, str)
        img=Image.open(input).convert('RGB')
        x=torch.unsqueeze(self.transforms(img),dim=0)
        y=self.net(x).data[0].numpy()
        if self.normalize:
            y /= np.sum(y)
        return y

if __name__ == "__main__":
    vgg_Ex = Vggnet_Ex()
    print(vgg_Ex.dimension)
    input = '../../data/style/0_0_001.png'
    hist = vgg_Ex.feature(input)
    input = '../../data/style/0_0_006.png'
    hist2 = vgg_Ex.feature(input)
    print(np.sum((hist - hist2) ** 2))

