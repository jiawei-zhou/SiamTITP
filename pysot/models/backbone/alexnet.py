from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.utils.misc1 import NestedTensor


class AlexNetLegacy(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNet.configs))
        super(AlexNetLegacy, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]

    def forward(self, x):
        x = self.features(x)
        return x


class AlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1):
        configs = list(map(lambda x: 3 if x == 3 else
                       int(x*width_mult), AlexNet.configs))
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            )
        self.feature_size = configs[5]

    def forward(self, x):
        mask = None
        if isinstance(x,NestedTensor):
            temp = x
            x = temp.tensors
            mask = temp.mask
        x = self.layer1(x)  # 29,29     61,61
        x = self.layer2(x)  # 12,12     28,28
        x = self.layer3(x)  # 10,10     26,26
        x = self.layer4(x)  # 8,8       24,24
        x = self.layer5(x)  # 6,6       22,22   C 512

        if mask is not None:
            mask = F.interpolate(mask[None].float(),size=x.shape[-2:]).to(torch.bool)[0]
            return x,mask
        else:
            return x
            
def alexnetlegacy(**kwargs):
    return AlexNetLegacy(**kwargs)


def alexnet(**kwargs):
    return AlexNet(**kwargs)

if __name__=='__main__':
    net = AlexNet(width_mult=2)
    x = torch.rand(1,3,127,127)
    x = net(x)