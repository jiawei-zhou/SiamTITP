# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from pysot.utils.misc1 import (NestedTensor,nested_tensor_from_tensor)
from pysot.models.backbone.alexnet import alexnetlegacy, alexnet
from pysot.models.backbone.mobile_v2 import mobilenetv2
from pysot.models.backbone.resnet_atrous import resnet18, resnet34, resnet50
import torch.nn.functional as F
import torch
BACKBONES = {
              'alexnetlegacy': alexnetlegacy,
              'mobilenetv2': mobilenetv2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': alexnet,
            }


def get_backbone(name, **kwargs):
    if 'resnet50' in name:
        name = 'resnet50'
    return BACKBONES[name](**kwargs)


