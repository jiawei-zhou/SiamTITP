# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamTITP"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 128

__C.TRAIN.SEARCH_SIZE = 256

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = '/home/zhoujiawei/tracking_model/SiamCAR-master/experiments/siamcar_r50/model_general.pth'

__C.TRAIN.LOG_DIR = '/home/zhoujiawei/tracking_model/SiamCAR-master/logs_SiamTITP_res50pretrained_30epoch_modified_head_and_attention_trackingnet_GOT_VID_LASOT_60w'

__C.TRAIN.SNAPSHOT_DIR = '/home/zhoujiawei/tracking_model/SiamCAR-master/snapshot_SiamTITP_res50pretrained_30epoch_modified_head_and_attention_trackingnet_GOT_VID_LASOT_60w'

__C.TRAIN.EPOCH = 30

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 128

__C.TRAIN.NUM_WORKERS = 48

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.CIOU_WEIGHT = 2.0

__C.TRAIN.L1_WEIGHT = 5.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

__C.TRAIN.LOSS_BCL_WEIGHT = [1,0.6]

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0


# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('LaSOT','VID', 'COCO', 'DET', 'YOUTUBEBB')
# __C.DATASET.NAMES = ('LaSOT',)

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = 'train_dataset/vid/crop511'          # VID dataset path
__C.DATASET.VID.ANNO = 'train_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

# __C.DATASET.YOUTUBEBB = CN()
# __C.DATASET.YOUTUBEBB.ROOT = 'train_dataset/yt_bb/crop511'  # YOUTUBEBB dataset path
# __C.DATASET.YOUTUBEBB.ANNO = 'train_dataset/yt_bb/train.json'
# __C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
# __C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'train_dataset/coco/crop511'         # COCO dataset path
__C.DATASET.COCO.ANNO = 'train_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

# __C.DATASET.DET = CN()
# __C.DATASET.DET.ROOT = 'train_dataset/det/crop511'           # DET dataset path
# __C.DATASET.DET.ANNO = 'train_dataset/det/train.json'
# __C.DATASET.DET.FRAME_RANGE = 1
# __C.DATASET.DET.NUM_USE = -1

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = 'train_dataset/got10k/crop511'         # GOT dataset path
__C.DATASET.GOT.ANNO = 'train_dataset/got10k/train.json'
__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 100000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = 'train_dataset/lasot/crop511'         # LaSOT dataset path
__C.DATASET.LaSOT.ANNO = 'train_dataset/lasot/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = -1   # -1

__C.DATASET.Trackingnet = CN()
__C.DATASET.Trackingnet.ROOT = 'train_dataset/trackingnet/crop511'         # trackingnet dataset path
__C.DATASET.Trackingnet.ANNO = 'train_dataset/trackingnet/train.json'
__C.DATASET.Trackingnet.FRAME_RANGE = 100
__C.DATASET.Trackingnet.NUM_USE = -1   # -1

__C.DATASET.VIDEOS_PER_EPOCH = 600000 # 600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

__C.BACKBONE.WIDTH_MULT = 2
# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'
__C.CAR.UP_FEAT_SIZE = 64
__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Template fusion options
# ------------------------------------------------------------------------ #
__C.TMfuse = CN()

__C.TMfuse.dim = 256

__C.TMfuse.dim_feedward = 2048

__C.TMfuse.fusion_layer_num = 2             # 2

__C.TMfuse.nhead = 8

# ------------------------------------------------------------------------ #
# Template fusion options
# ------------------------------------------------------------------------ #
__C.POS = CN()
__C.POS.feats_num = 128
# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.UP_FEAT_SIZE = 64

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44

__C.HP_TRACK_NUM = CN()

__C.HP_TRACK_NUM.SatSOT = [50,30] # trajeactory and tf  # SV248S 60 

__C.HP_TRACK_NUM.SV248S= [60,50]   # SV248S 50

__C.HP_TRACK_NUM.OOTB= [40,50]

__C.HP_TRACK_NUM.UAV123= [40,50]

__C.HP_TRACK_NUM.LaSOT= [40,50]
# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB100 = [0.35, 0.2, 0.45]

__C.HP_SEARCH.GOT10K = [0.7, 0.06, 0.1]

__C.HP_SEARCH.SatSOT = [0.35, 0.2, 0.45]
# __C.HP_SEARCH.SatSOT = [1, 1, 1]
__C.HP_SEARCH.SV248S = [0.35, 0.2, 0.45]

__C.HP_SEARCH.VISO = [0.35, 0.2, 0.45]

__C.HP_SEARCH.OOTB = [0.35, 0.2, 0.45]

__C.HP_SEARCH.UAV123 = [0.4, 0.2, 0.3]

__C.HP_SEARCH.LaSOT = [0.33, 0.04, 0.3]
