import os 
import json
import numpy as np
import pandas as pd
import random
def gen_json_file():
    path ='Your_dataset_path/OOTB'    # need to modified path
    videos = sorted(next(os.walk(path))[1])
    js = {}
    for video in videos:
        if video == 'anno':
            continue
        video_name = video
        video_dir = video
        gt_path = os.path.join(path,video,'new_groundtruth_hbb.txt')    # need to convert the rotation bounding box to horizontal bounding box
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
        # save json information
        js[video_name] = {}
        js[video_name]['video_dir'] = video_dir
        js[video_name]['init_rect'] = init_bbox
        js[video_name]['img_names'] = img_path_list
        js[video_name]['gt_rect'] = gt_bbox
        js[video_name]['attr'] = video_attr

    json.dump(js, open('Your_dataset_path/OOTB.json', 'w'), indent=4, sort_keys=True)         
gen_json_file()
