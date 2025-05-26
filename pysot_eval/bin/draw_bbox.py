# Copyright (c) SenseTime. All Rights Reserved.

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
env_path = os.path.join(os.path.dirname(__file__),'..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pysot.visualization.draw_utils import color_line
from pysot.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')

sv_dir = os.path.join(os.path.dirname(__file__),'..')
dataset_dir = 'Your_dataset_path' # SatSOT SV248S
datset_name = ''    # SV248S OOTB SatSOT
result_dir = 'Your_result_path'    
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default=datset_name,
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')
parser.add_argument('--dataset_dir', type=str, default=dataset_dir,
        help='dataset path')
parser.add_argument('--save_dir', default = sv_dir , type=str, 
        help="result path")

args = parser.parse_args()
video_names = []
tracker_names = ['SiamTITP(Ours)']
color = color_line(tracker_names) 
for key, v in color.items():
    temp = (v[2]* 255,v[1]* 255,v[0]* 255)
    color[key] = temp
def main():
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=args.dataset_dir,
                                            load_img=False)
    for v_idx, video in enumerate(dataset):
        if len(video_names) != 0:
            # test one special video
            if video.name not in video_names:
                continue
        pred_bboxes = {}
        video_name = video.name
        for track in  tracker_names:
            path = os.path.join(result_dir,track,video_name,video_name +'.txt')
            split = ['\t',',']
            try:
                pred_bboxes[track] = np.loadtxt(path,delimiter=split[0]).astype(np.int64)
            except:
                pred_bboxes[track] = np.loadtxt(path,delimiter=split[1]).astype(np.int64)
        for idx, (img, gt_bbox) in enumerate(video):
            gt_bbox = list(map(int,gt_bbox))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.rectangle(img,(gt_bbox[0],gt_bbox[1]),(gt_bbox[0]+gt_bbox[2],gt_bbox[1]+gt_bbox[3]),(0,215,255),1)
            for i,track in enumerate(tracker_names):
                img = cv2.rectangle(img,(pred_bboxes[track][idx][0],pred_bboxes[track][idx][1]), 
                                    (pred_bboxes[track][idx][0]+pred_bboxes[track][idx][2], 
                                     pred_bboxes[track][idx][1]+pred_bboxes[track][idx][3]),color[track],1)
            visual_result_path = os.path.join(args.save_dir,'bbox_result_failed_2025.05.06',args.dataset,video_name)
            if not os.path.exists(visual_result_path):
                os.makedirs(visual_result_path)
            visual_result_file = os.path.join(visual_result_path,'{:03d}'.format(idx) + '.png')
            
            cv2.imwrite(visual_result_file,img)
        print(video_name,'has been done')
        
if __name__ == '__main__':
    main()
