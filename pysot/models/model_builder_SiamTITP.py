# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from pysot.models.template_fuse import TMfuse
from pysot.models.postion_embeding import PositionEmbeddingSine
from pysot.core.config import cfg
from pysot.models.loss_TITP import make_SiamTITP_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.SiamTITP_head import SiamTITPHead
from pysot.models.neck import get_neck
from pysot.utils.xcorr import xcorr_depthwise
from pysot.utils.misc1 import (NestedTensor,nested_tensor_from_tensor)

class SiamTITP(nn.Module):
    def __init__(self):
        super(SiamTITP, self).__init__()

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
        self.car_head = SiamTITPHead(cfg,256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_SiamTITP_loss_evaluator(cfg)

        self.down_ = nn.ConvTranspose2d(256 * 2, 256, 1, 1)

        self.feats_down = nn.ConvTranspose2d(256*2,256,1,1)

    def template(self, z):
        if not isinstance(z,NestedTensor):
            z = nested_tensor_from_tensor(z) 
        zf,mask_zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf,mask_zf = self.neck(zf,mask_zf)
        # template self attention
        zf_fuse = self.feats_down(torch.cat((zf[1],zf[2]),dim=1))
        pos_1 = self.pos_embed(zf_fuse,mask_zf)
        zf_ = self.tmfuse(src_template=zf_fuse,src_dny_template=zf_fuse,
                          pos_tem=pos_1,pos_dny_tem=pos_1,
                          mask_tem=mask_zf,mask_dny_tem=mask_zf)
        self.mask = mask_zf
        # get initial zf two elements
        self.zf = torch.cat((zf[0].unsqueeze(0),zf_.unsqueeze(0)),dim=0)
        self.pos = pos_1

    def track_template(self,z):
        if not isinstance(z,NestedTensor):
            z = nested_tensor_from_tensor(z)
        zf,mask_zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf,mask_zf = self.neck(zf,mask_zf)
        # current template with pre template to fues
        zf_fuse = self.feats_down(torch.cat((zf[1],zf[2]),dim=1))
        zf_ = self.tmfuse(src_template=self.zf[1],src_dny_template=zf_fuse,
                          pos_tem=self.pos,pos_dny_tem=self.pos,
                          mask_tem=self.mask,mask_dny_tem=mask_zf)
        self.mask = mask_zf
        # save zf[0] and update zf[1],zf[2]
        self.zf = torch.cat((self.zf[0].unsqueeze(0),zf_.unsqueeze(0)),dim=0)

    def track(self, x,):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        xf_ = self.feats_down(torch.cat((xf[1],xf[2]),dim=1))
        features_new = self.xcorr_depthwise(xf_,self.zf[1])
        features = torch.cat([features,features_new],dim=1)
        features = self.down_(features)
        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        dny_template = data['dny_template'].cuda()
        search = data['search'].cuda()
        label_loc = data['bbox'].cuda() # x,y,x,y

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
        # adjust features 4,5 layer
        zf1_cat = self.feats_down(torch.cat((zf1[1],zf1[2]),dim=1))   # b,256,h,w  
        zf2_cat = self.feats_down(torch.cat((zf2[1],zf2[2]),dim=1))
        xf_ = self.feats_down(torch.cat((xf[1],xf[2]),dim=1))
        # postion embeding and attention model

        pos_1 = self.pos_embed(zf1_cat,mask_zf1)     # 位置编码
        pos_2 = self.pos_embed(zf2_cat,mask_zf2)

        zf_ = self.tmfuse(src_template=zf1_cat,src_dny_template=zf2_cat,
                          pos_tem=pos_1,pos_dny_tem=pos_2,
                          mask_tem=mask_zf1,mask_dny_tem=mask_zf2)   # 得到了融合4,5层和动态模板的模板
        # deep of corss relation
        features = self.xcorr_depthwise(xf[0],zf1[0])
        features_new = self.xcorr_depthwise(xf_,zf_)
        features = torch.cat([features,features_new],1)
        features = self.down_(features)

        pr_bbox = self.SiamTITP_head(features)   # pr_bbox cx,cy,w,h
        total_loss, ciou_loss,l1_loss,iou_loss  = self.loss_evaluator(
            pr_bbox,label_loc
        )

        # get loss
        outputs = {}
        outputs['total_loss'] = total_loss
        outputs['ciou_loss'] = ciou_loss
        outputs['l1_loss'] = l1_loss
        outputs['iou_loss'] = iou_loss
        return outputs