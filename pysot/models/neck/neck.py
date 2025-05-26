# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            if x.size(3) > 6:
                l = 4
                r = l + 7
                x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features,mask=None):
        
        def cut_window(x):
            if x.size(1) < 20:
                l = 4
                r = l + 7
                x = x[:,l:r, l:r]
            return x
        
        if self.num == 1:
            features = self.downsample(features)
            if mask is not None:
                if features.shape[2] != mask.shape[1]:
                    mask = cut_window(mask)
                return features,mask
            else:
                return features
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]).contiguous())
            if mask is not None:
                if out[0].shape[2] != mask.shape[1]:
                    mask = cut_window(mask)
                return out,mask
            else:
                return out
