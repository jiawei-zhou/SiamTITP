from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
from skimage.measure import label, regionprops
from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip

class SiamTITPracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamTITPracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()
        self.occlusion_num = 0

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
       
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))
        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.track_trajce = np.array([bbox[0]+(bbox[2])/2,bbox[1]+(bbox[3])/2]).reshape(1,2)
        self.model.template(z_crop)

    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)  # 计算193与255中心坐标对齐两边相差的值
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])   # 得到255大小中目标位置
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.  # 得到对于255中心点位置的偏移量
        return disp

    def coarse_location(self, hp_score_up, p_score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape)    # 找到hp_score_up中的最大值的索引
        max_r = int(round(max_r_up_hp / scale_score))   # 找到最大值在25大小中的位置
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)   # 裁剪位置，大于0且小于25
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]    # 由hp_score_up的最大位置由回归图得到bbox_region，ltrb
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)    # 设置的超参数  最小边距
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)    # 最大边距
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0)
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0)

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        mask = np.zeros_like(p_score_up)    # 创建与p_score_up大小相同的全零数组
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1  #[row,clu] mask得到目标掩膜
        p_score_up = p_score_up * mask
        return p_score_up

    def getCenter(self,hp_score_up, p_score_up, scale_score,lrtbs):
        # corse location
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, lrtbs)    # 得到目标掩膜
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)    # 得到中心位置
        disp = self.accurate_location(max_r_up,max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy

    def track(self, img, hp,idx):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        self.num = 0
        pred_x = None
        pred_y = None
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy()
        cen = (cen - cen.min()) / cen.ptp()     # .ptp()得到峰值，最大值减最小值
        cen = cen.squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1

        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_score = penalty * cls * cen
        if cfg.TRACK.hanming:
            hp_score = p_score*(1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_score = p_score
        if (self.peak_range(hp_score) or (len(self.peak_num(hp_score)) > 1)):
            self.num += 1
            if idx > cfg.TRACK.POLY_NUM:
                x = self.track_trajce[:,0][-cfg.TRACK.POLY_NUM:]
                y = self.track_trajce[:,1][-cfg.TRACK.POLY_NUM:]
                frame_num = np.arange(0,cfg.TRACK.POLY_NUM)
                func_x = np.polyfit(frame_num, x, 1)
                func_y = np.polyfit(frame_num, y, 1)
                pred_x = np.polyval(func_x,cfg.TRACK.POLY_NUM + 1)
                pred_y = np.polyval(func_y,cfg.TRACK.POLY_NUM + 1)
            
        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs,(1,2,0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, lrtbs)
        # get w h
        ave_w = (lrtbs_up[max_r_up,max_c_up,0] + lrtbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up,max_c_up,1] + lrtbs_up[max_r_up,max_c_up,3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))

        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']

        new_width = lr*ave_w + (1-lr)*self.size[0]
        new_height = lr*ave_h + (1-lr)*self.size[1]
        # clip boundary
        if pred_x is not None:
            cx = bbox_clip(pred_x,1,img.shape[1])
            cy = bbox_clip(pred_y,1,img.shape[0])
        else:
            cx = bbox_clip(new_cx,1,img.shape[1])
            cy = bbox_clip(new_cy,1,img.shape[0])
            
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])
        left_x = bbox_clip(cx - width / 2,1,img.shape[1])
        left_y = bbox_clip(cy - height / 2,1,img.shape[0])
        self.center_pos = np.array([cx, cy]).reshape(1,2)
        self.track_trajce = np.concatenate((self.track_trajce,self.center_pos),axis=0)
        self.size = np.array([width, height])
        bbox = [left_x,
                left_y,
                width,
                height]
        self.center_pos = self.center_pos.squeeze()  
        if self.num == 0 and (idx+1) % cfg.TRACK.TF_NUM == 0:
            z_crop = self.get_subwindow(img, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        round(s_z), self.channel_average)
            self.model.track_template(z_crop)
        return {
                'bbox': bbox
            }

    def peak_num(self,hp_score):
        high_value_group = cv2.resize(hp_score,(255,255))>(np.max(hp_score)-0.1)
        # 标记连通组件
        label_image = label(high_value_group)

        # 获取连通组件的属性
        regions = regionprops(label_image)
        
        # 获取高值群的中心坐标
        high_value_centers = [tuple(map(int, region.centroid)) for region in regions]
        return(high_value_centers)
    
    def peak_range(self,data):
        data = cv2.resize(data,(255,255))
        # peak = np.max(data)
        peak_area = np.where(data > np.max(data)-0.1)[0].base   # 0是行坐标，1是列坐标
        y = np.max(peak_area[:,0]) - np.min(peak_area[:,0])
        x = np.max(peak_area[:,1]) - np.min(peak_area[:,1])
        peak_proportion = x * y
        target_proportion = self.size[0] * self.size[1] * self.scale_z**2 / 4
        return peak_proportion > target_proportion
