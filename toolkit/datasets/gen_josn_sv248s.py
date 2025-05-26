# import os 
# import json
# import numpy as np
# import pandas as pd
# path ='/home/zhoujiawei/satellite_video_datasets/SV248S'
# files = sorted(next(os.walk(path))[1])
# js = {}
# for file in files:
#     videos = sorted(os.listdir(os.path.join(path,file,'sequences')))
#     for video in videos:
#         video_info_path = os.path.join(path,file,'annotations',video + '.abs')
#         with open(video_info_path, 'r') as f:
#         # 读取CSV内容
#             csv_content = f.read()
#         video_info = json.loads(csv_content)
#         if len(js):
#             num = 1
#             video_name = video_info['details']['class_name']
#             video_name = video_name.replace('-','_')
#             if video_name == 'car':
#                 video_names = [item for item in js.keys() if 'car_large' not in item]
#             else:
#                 video_names = list(js.keys())
#             for key in video_names:
#                 if video_name in key:
#                     num += 1
#             video_name = video_name + '_{:03d}'.format(num)
#         else:
#             video_name = video_info['details']['class_name'] + '_{:03d}'.format(1)
#             video_name = video_name.replace('-','_')
#         video_dir = video_info['source_info']['video_id'] + '/' + video_info['source_info']['seq_id']
#         video_init_rect = list(map(float,np.array(video_info['details']['init_rect']).reshape(4,)))

#         video_img_names = []
#         for i in range(int(video_info['details']['length'])):
#             video_img_names.append(os.path.join(file,'sequences',video,'{:06}'.format(i+1) + '.tiff'))

#         video_gtrect_path = os.path.join(path,file,'annotations',video + '.rect')
#         video_gtrect = list(np.loadtxt(video_gtrect_path,delimiter=',').astype(np.float32))
#         gts = []
#         for gt in video_gtrect:
#             gts.append(list(map(float,list(gt))))
#         video_attr_path = os.path.join(path,file,'annotations',video + '.attr')
#         video_attr_mask = np.loadtxt(video_attr_path,delimiter=',').astype(np.float32)
#         attrs = ['STO','LTO','DS','IV','BCH','SM','ND','CO','BCL','IPR']
#         video_attr = []
#         for mask,attr in zip(video_attr_mask,attrs):
#             if mask:
#                 video_attr.append(attr)
#         # 保存json
#         js[video_name] = {}
#         js[video_name]['video_dir'] = video_dir
#         js[video_name]['init_rect'] = video_init_rect
#         js[video_name]['img_names'] = video_img_names
#         js[video_name]['gt_rect'] = gts
#         js[video_name]['attr'] = video_attr

# json.dump(js, open('/home/zhoujiawei/satellite_video_datasets/SV248S/SV248S.json', 'w'), indent=4, sort_keys=True)

import os 
import json
import numpy as np
import pandas as pd
path ='/home/zhoujiawei/satellite_video_datasets/SV248S'
files = sorted(next(os.walk(path))[1])
js = {}
for file in files:
    videos = sorted(os.listdir(os.path.join(path,file,'sequences')))
    for video in videos:
        video_info_path = os.path.join(path,file,'annotations',video + '.abs')
        with open(video_info_path, 'r') as f:
        # 读取CSV内容
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
        # 保存json
        js[video_name] = {}
        js[video_name]['video_dir'] = video_dir
        js[video_name]['init_rect'] = video_init_rect
        js[video_name]['img_names'] = video_img_names
        js[video_name]['gt_rect'] = gts
        js[video_name]['attr'] = video_attr

json.dump(js, open('/home/zhoujiawei/satellite_video_datasets/SV248S/SV248S_new.json', 'w'), indent=4, sort_keys=True)