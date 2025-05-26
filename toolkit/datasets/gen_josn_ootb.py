import os 
import json
import numpy as np
import pandas as pd
import random
def random_choices_gt_file():
    path ='/home/zhoujiawei/satellite_video_datasets/OOTB'
    videos = sorted(next(os.walk(path))[1])
    js = {}
    for video in videos:
        if video == 'anno':
            continue
        video_name = video
        video_dir = video
        gt_path = os.path.join(path,video,'new_groundtruth_hbb.txt')
        # init_rect_list = []
        # gt_rect_list = []
        gt_bbox = np.loadtxt(gt_path,delimiter=',').astype(float)
        gt_bbox = [list(bbox) for bbox in gt_bbox]
        init_bbox = gt_bbox[0]
        img_path_list = []
        img_path = os.path.join(path,video,'img')
        img_names = sorted(os.listdir(img_path))
        for img_name in img_names:
            img_path_list.append(os.path.join(video,'img',img_name))
        # 'DEF', 'IPR', 'PO', 'FO', 'IV', 'MB', 'BC', 'OON', 'SA', 'LT', 'IM', 'AM'
        video_attr_path = os.path.join(path,'anno',video + '.txt')
        video_attr_mask = np.loadtxt(video_attr_path,delimiter=',').astype(np.float32)[4:]
        attrs = ['DEF', 'IPR', 'PO', 'FO', 'IV', 'MB', 'BC', 'OON', 'SA', 'LT', 'IM', 'AM']
        video_attr = []
        for mask,attr in zip(video_attr_mask,attrs):
            if mask:
                video_attr.append(attr)
        # 保存json
        js[video_name] = {}
        js[video_name]['video_dir'] = video_dir
        js[video_name]['init_rect'] = init_bbox
        js[video_name]['img_names'] = img_path_list
        js[video_name]['gt_rect'] = gt_bbox
        js[video_name]['attr'] = video_attr

    json.dump(js, open('/home/zhoujiawei/satellite_video_datasets/OOTB/OOTB_new.json', 'w'), indent=4, sort_keys=True)
# def check_gtfiles():
#     path ='/home/zhoujiawei/satellite_video_datasets/VISO/SOT/sot'
#     classes = sorted(next(os.walk(path))[1])
#     js = {}
#     for classe in classes:
#         videos = sorted(os.listdir(os.path.join(path,classe)))
#         for video in videos:
#             video_name = classe + '_' + video
#             video_dir = video
#             attr = None
#             gt_path = os.path.join(path,classe,video,'gt')
#             init_rect_list = []
#             gt_rect_list = []
#             gt_files = os.listdir(gt_path)
#             new_gt_path = os.path.join(path,classe,video,'new_gt')
#             if not os.path.exists(new_gt_path):
#                 os.makedirs(new_gt_path)
#             for gt_file in gt_files:
#                 gt_file_path = os.path.join(gt_path,gt_file)
#                 new_gt_file_path = os.path.join(new_gt_path,gt_file)
#                 file_data = ""
#                 with open(gt_file_path,'r') as f:
#                     for line in f:
#                         first_obj = line.split(',')[0]
#                         if first_obj == '':
#                             line = line.replace(',','',1)
#                         file_data += line
#                 with open(new_gt_file_path,'w') as f:
#                     f.write(file_data)
# check_gtfiles()              
random_choices_gt_file()