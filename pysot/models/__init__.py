# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .model_builder_SiamTITP import SiamTITP
MODEBUILDER = {
    'SiamTITP_resnet50':SiamTITP,
}

def get_modelbuilder(name,**Kwargs):
    return MODEBUILDER[name](**Kwargs)