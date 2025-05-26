import os 
import json
import numpy as np
import pandas as pd
path ='Your_dataset_path/SV248S'    # the dataset path
files = sorted(next(os.walk(path))[1])
js = {}
for file in files:
    videos = sorted(os.listdir(os.path.join(path,file,'sequences')))
    for video in videos:
        video_info_path = os.path.join(path,file,'annotations',video + '.abs')
        with open(video_info_path, 'r') as f:
        # read csv information
            csv_content = f.read()
        video_info = json.loads(csv_content)
        video_name = video_info['source_info']['video_id'] + '_' +video_info['source_info']['seq_id']

        video_dir = video_info['source_info']['video_id'] + '/' + video_info['source_info']['seq_id']
        video_init_rect = list(map(float,np.array(video_info['details']['init_rect']).reshape(4,)))

        video_img_names = []
        for i in range(int(video_info['details']['length'])):
            video_img_names.append(os.path.join(file,'sequences',video,'{:06}'.format(i+1) + '.tiff'))

        video_gtrect_path = os.path.join(path,file,'annotations',video + '.rect')
        video_gtrect = list(np.loadtxt(video_gtrect_path,delimiter=',').astype(np.float32))
        gts = []
        for gt in video_gtrect:
            gts.append(list(map(float,list(gt))))
        video_attr_path = os.path.join(path,file,'annotations',video + '.attr')
        video_attr_mask = np.loadtxt(video_attr_path,delimiter=',').astype(np.float32)
        attrs = ['STO','LTO','DS','IV','BCH','SM','ND','CO','BCL','IPR']
        video_attr = []
        for mask,attr in zip(video_attr_mask,attrs):
            if mask:
                video_attr.append(attr)
        # save information
        js[video_name] = {}
        js[video_name]['video_dir'] = video_dir
        js[video_name]['init_rect'] = video_init_rect
        js[video_name]['img_names'] = video_img_names
        js[video_name]['gt_rect'] = gts
        js[video_name]['attr'] = video_attr

json.dump(js, open('your_dataset_path/sv248s.json', 'w'), indent=4, sort_keys=True)
