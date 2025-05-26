# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pysot.models.template_fuse import TMfuse
from pysot.models.postion_embeding import PositionEmbeddingSine
from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from pysot.utils.misc1 import (NestedTensor,nested_tensor_from_tensor)
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
class ModelBuilder_Alexnet(nn.Module):
    def __init__(self):
        super(ModelBuilder_Alexnet, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build template fuse
        self.tmfuse = TMfuse(cfg.TMfuse.dim,cfg.TMfuse.nhead,
                             cfg.TMfuse.dim_feedward,cfg.TMfuse.fusion_layer_num)

        # build postion embeding
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=cfg.POS.feats_num,normalize=True)
        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

    def template(self, z):
        if not isinstance(z,NestedTensor):
            z = nested_tensor_from_tensor(z) 
        zf,mask_zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf,mask_zf = self.neck(zf,mask_zf)
        # template self attention
        pos_1 = self.pos_embed(zf,mask_zf)
        zf_ = self.tmfuse(src_template=zf,src_dny_template=zf,
                          pos_tem=pos_1,pos_dny_tem=pos_1,
                          mask_tem=mask_zf,mask_dny_tem=mask_zf)
        self.mask = mask_zf
        # get initial zf two elements
        self.zf = zf_
        self.pos = pos_1
        return self.zf

    def track_template(self,z):
        if not isinstance(z,NestedTensor):
            z = nested_tensor_from_tensor(z)
        zf,mask_zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf,mask_zf = self.neck(zf,mask_zf)
        # current template with pre template to fues
        zf_ = self.tmfuse(src_template=self.zf,src_dny_template=zf,
                          pos_tem=self.pos,pos_dny_tem=self.pos,
                          mask_tem=self.mask,mask_dny_tem=mask_zf)
        self.mask = mask_zf
        # save zf[0] and update zf[1],zf[2]
        self.zf = zf_

    def track(self, x,):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf,self.zf)
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
    
    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        dny_template = data['dny_template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        if not isinstance(template,NestedTensor):
            zf1 = nested_tensor_from_tensor(template)    # zf1_cat b,256,h,w
        if not isinstance(dny_template,NestedTensor):
            zf2 = nested_tensor_from_tensor(dny_template)
        zf1,mask_zf1 = self.backbone(zf1)   
        zf2,mask_zf2 = self.backbone(zf2)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf1,mask_zf1 = self.neck(zf1,mask_zf1)
            zf2,mask_zf2 = self.neck(zf2,mask_zf2)
            xf = self.neck(xf)
        # # adjust features 4,5 layer
        # zf1_cat = self.feats_down(torch.cat((zf1[1],zf1[2]),dim=1))   # b,256,h,w  
        # zf2_cat = self.feats_down(torch.cat((zf2[1],zf2[2]),dim=1))
        # xf_ = self.feats_down(torch.cat((xf[1],xf[2]),dim=1))
        # postion embeding and attention model

        pos_1 = self.pos_embed(zf1,mask_zf1)     # 位置编码
        pos_2 = self.pos_embed(zf2,mask_zf2)

        zf_ = self.tmfuse(src_template=zf1,src_dny_template=zf2,
                          pos_tem=pos_1,pos_dny_tem=pos_2,
                          mask_tem=mask_zf1,mask_dny_tem=mask_zf2)   # 得到了动态模板
        # deep of corss relation
        features = self.xcorr_depthwise(xf,zf_)
   
        cls, loc, cen = self.car_head(features)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)    # 生成网格
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs

    def draw(self,data):
        font = FontProperties()
        #作图阶段
        fig = plt.figure()
        #定义画布为1*1个划分，并在第1个位置上进行作图
        # ax = fig.add_subplot(111)
        ax = plt.gca()
        #定义横纵坐标的刻度间隔
        x_major_locator = MultipleLocator(30)
        y_major_locator = MultipleLocator(30)

        ax.xaxis.set_major_locator(x_major_locator)
        ax.yaxis.set_major_locator(y_major_locator)
        #作图并选择热图的颜色填充风格，这里选择hot
        im = ax.imshow(data, cmap='rainbow')
        #增加右侧的颜色刻度条
        plt.colorbar(im)
        #增加标题
        plt.title("correlation map", fontproperties=font)
        plt.tight_layout()
        #show
        plt.savefig('heatmap.png')