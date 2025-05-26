# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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

from pysot.core.config import cfg
from pysot.tracker.siamTITP_tracker import SiamTITPracker       
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models import get_modelbuilder

from toolkit.datasets import DatasetFactory

parser = argparse.ArgumentParser(description='siamcar tracking')

file_path = os.path.join(os.path.dirname(__file__),'..')
snapshot = '/home/zhoujiawei/tracking_model/SiamTITP/experiments/SiamTITP_r50/SiamTITP_res50.pth'   # weighted file path
config = '/home/zhoujiawei/tracking_model/SiamTITP/experiments/SiamTITP_r50/config.yaml'    # config file path 
dataset_dir = '/home/zhoujiawei/satellite_video_datasets/SatSOT'   # SatSOT SV248S /home/zhoujiawei/train_dataset/UAV123 '/home/zhoujiawei/satellite_video_datasets/SatSOT'
datasets_name = 'SatSOT'
model_name = 'SiamTITP_test'    # want saved tracker name

parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default=datasets_name,
        help='datasets')#OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true',default=False,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default=snapshot,
        help='snapshot of models to eval')
parser.add_argument('--config', type=str, default=config,
        help='config file')
parser.add_argument('--dataset_dir', type=str, default=dataset_dir,
        help='dataset path')
parser.add_argument('--save_dir', default = file_path , type=str, 
        help="result path")

args = parser.parse_args()

torch.set_num_threads(1)
# 2024 0716 更改了 dataset中sv248s的数据集评估部分

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # load config
    cfg.merge_from_file(args.config)

    # hp_search
    params = getattr(cfg.HP_SEARCH,args.dataset)
    hp = {'lr': params[0], 'penalty_k':params[1], 'window_lr':params[2]}

    track_params = getattr(cfg.HP_TRACK_NUM,args.dataset)
    cfg.TRACK.POLY_NUM = track_params[0]
    cfg.TRACK.TF_NUM = track_params[1]
    print('POLY_NUM:',track_params[0],'TF_NUM:',track_params[1])
    model = get_modelbuilder(cfg.META_ARC)

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamTITPracker(model, cfg.TRACK)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=args.dataset_dir,
                                            load_img=False)

    print(args.snapshot.split('/')[-2] + '_' + str(hp['lr']) + '_' + str(hp['penalty_k']) + '_' + str(hp['window_lr']))

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        video_name = video.name
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                # gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox)
                pred_bbox = gt_bbox
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img,hp,idx)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                # print('video:{} idx:{} predbox:{} groundtruth:{}'.format(video_name,idx,list(map(int,pred_bbox)),gt_bbox))
                # scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            end = cv2.getTickCount()
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            print('({:3d})Video: {:12s} idx :{:3d}  Time: {:2.4f}s, compelet'.format(v_idx+1,
                video_name, idx+1, (end - tic)/cv2.getTickFrequency()))
        toc /= cv2.getTickFrequency()

        # save reslut
        # model_path = os.path.join(args.save_dir,'test_results', args.dataset, model_name,video_name)
        model_path = os.path.join(args.save_dir,'results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video_name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')

        # result_path1 = os.path.join(model_path, '{}_score.txt'.format(video_name))
        # with open(result_path1, 'w') as f:
        #     for id, x in enumerate(scores):
        #         if x == None:
        #             continue
        #         f.write('id:{:3d} best score{:1.4f}'.format(id,x) + '\n')

        result_path_time = os.path.join(model_path, '{}_time.txt'.format(video_name))
        with open(result_path_time, 'w') as f:
            for id, x in enumerate(track_times):
                f.write( str(x) + '\n')

        print('({:d})   Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video_name, toc, (idx+1) / toc))
        # save results
    #     model_path = os.path.join('results', args.dataset, model_name)
    #     if not os.path.isdir(model_path):
    #         os.makedirs(model_path)
    #     result_path = os.path.join(model_path, '{}.txt'.format(video.name))
    #     with open(result_path, 'w') as f:
    #         for x in pred_bboxes:
    #             f.write(','.join([str(i) for i in x])+'\n')
    #     print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
    #         v_idx+1, video.name, toc, idx / toc))
    # os.chdir(model_path)
    # save_file = '../%s' % dataset
    # shutil.make_archive(save_file, 'zip')
    # print('Records saved at', save_file + '.zip')
    
if __name__ == '__main__':
    main()
