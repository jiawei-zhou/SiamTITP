import os
import sys
import time
import argparse
import functools
env_path = os.path.join(os.path.dirname(__file__),'..')
if env_path not in sys.path:
    sys.path.append(env_path)

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from pysot.datasets import OTBDataset, UAVDataset, LaSOTDataset, VOTDataset, NFSDataset, VOTLTDataset,SatSOTDataset,SV248SDataset,OOTBDataset
from pysot.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, EAOBenchmark, F1Benchmark
from pysot.visualization import draw_success_precision, draw_eao, draw_f1,draw_radar_plot
from pysot.visualization.draw_utils import color_line


dataset_dir = '/home/zhoujiawei/satellite_video_datasets/SatSOT' #'/home/zhoujiawei/satellite_video_datasets/SV248S'    /home/zhoujiawei/satellite_video_datasets/OOTB
# /home/zhoujiawei/satellite_video_datasets/SatSOT /home/zhoujiawei/train_dataset/UAV123
# /home/zhoujiawei/tracking_model/siamban_master/testing_dataset/OTB-100
# dataset_dir = r"F:\Linux2Win10\OTB1001\OTB100"    SatSOT  SV248S  UAV123
dataset = 'SatSOT'

# tracker_results_dir = r"F:\Linux2Win10\OTB1001\OTB100_backup"
tracker_results_dir = '/home/zhoujiawei/tracking_model/SiamTITP/results/SatSOT'
# 'MDNet',
# trackers = [ 'SiamCAR', 'CFNet', 'DaSiamRPN', 'GradNet', 'SRDCF', 'fDSST', 'DeepSRDCF', 'SiamRPN', 'SiamDWfc', 'Staple']
trackers = ['SiamTITP_test']
# trackers = ['SiamTITP_modified_attention_GOT_VID_LASOT_50w_ckpt10','SiamTITP_modified_attention_GOT_VID_LASOT_50w_ckpt20','SiamTITP_modified_attention_GOT_VID_LASOT_50w_ckpt18','SiamTITP_modified_attention_GOT_VID_LASOT_50w_ckpt17','SiamTITP_modified_attention_GOT_VID_LASOT_50w_ckpt16']
# trackers = ['SiamTITP(Ours)','SiamTITP_only_TP','SiamTITP_only_TI','SiamCAR(baseline)']
# trackers = ['SiamTITP_30epoch_modified_attention_COCO_GOT_VID_LASOT_60w_ckpt30','SiamTITP_30epoch_modified_attention_COCO_GOT_VID_LASOT_60w_ckpt28','SiamTITP_30epoch_modified_attention_COCO_GOT_VID_LASOT_60w_ckpt29']
color_line(trackers=trackers)
num = 1
vis = False
if __name__ == '__main__':
    #step1、创建一个解析器——创建 ArgumentParser() 对象
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')

    #step2、添加参数——调用 add_argument() 方法添加参数
    parser.add_argument('--dataset_dir', type=str, default=dataset_dir, help='dataset root directory')#数据集根目录
    parser.add_argument('--dataset', type=str, default=dataset, help='dataset name')#数据集名称
    parser.add_argument('--tracker_result_dir', type=str, default=tracker_results_dir, help='tracker result root')#tracker 结果路径
    parser.add_argument('--trackers', default=trackers, nargs='+',help='Trackers name')#nargs='+' 设置一个或多个参数， 同时评估多个tracker
    parser.add_argument('--vis', dest='vis',default=vis, action='store_true')#可视化，为真时绘图
    parser.add_argument('--show_video_level', dest='show_video_level', action='store_true')
    parser.add_argument('--num', type=int, help='number of processes to eval', default=16)#线程数目，有默认值 1，不是很重要

    #step3、解析参数——使用 parse_args() 解析添加的参数
    args = parser.parse_args()

    tracker_dir = args.tracker_result_dir#跟踪结果目录
    trackers = args.trackers#跟踪器名称
    root = args.dataset_dir#数据集根目录

    assert len(trackers) > 0#断言：至少有一个跟踪器
    args.num = min(args.num, len(trackers))#每个跟踪器配备一个进程，自适应跟踪器数量 与 进程数


    #OTB数据集上评价结果
    if 'OTB' == args.dataset:
        dataset = OTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        #是否绘制跟踪结果
        if args.vis:
            radar_plot = {}
            for attr, videos in dataset.attr.items():
                success,precision,EIoU=draw_success_precision(success_ret,
                                                            name=dataset.name,
                                                            videos=videos,
                                                            attr=attr,
                                                            precision_ret=precision_ret,
                                                            dataset_name=args.dataset)
                radar_plot[attr]=[success,precision,EIoU]
            draw_radar_plot(radar_plot)
    elif 'OOTB' in args.dataset:
        dataset = OOTBDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        EIoU_success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_EIou,
                trackers), desc='eval EIoU', total=len(trackers), ncols=100):
                EIoU_success_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,EIoU_success_ret,
                show_video_level=args.show_video_level)
        #是否绘制跟踪结果
        if args.vis:
            radar_plot = {}
            for attr, videos in dataset.attr.items():
                success,precision,EIoU=draw_success_precision(success_ret,
                            name=dataset.name,
                            videos=videos,
                            attr=attr,
                            bold_name='SiamDTF(Ours)',
                            precision_ret=precision_ret,
                            EIoU_success_ret = EIoU_success_ret,
                            dataset_name=args.dataset)
                radar_plot[attr] = {}
                radar_plot[attr]['success']=success;radar_plot[attr]['precision']=precision
                radar_plot[attr]['EIoU']=EIoU;radar_plot[attr]['num'] = len(videos)
            draw_radar_plot(radar_plot,trackers,args.dataset)

    elif 'SatSOT' in args.dataset:
        dataset = SatSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        EIoU_success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_EIou,
                trackers), desc='eval EIoU', total=len(trackers), ncols=100):
                EIoU_success_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,EIoU_success_ret,
                show_video_level=args.show_video_level)
        #是否绘制跟踪结果
        if args.vis:
            radar_plot = {}
            for attr, videos in dataset.attr.items():
                success,precision,EIoU=draw_success_precision(success_ret,
                            name=dataset.name,
                            videos=videos,
                            attr=attr,
                            bold_name='SiamDTF(Ours)',
                            precision_ret=precision_ret,
                            EIoU_success_ret = EIoU_success_ret,
                            dataset_name=args.dataset)
                radar_plot[attr] = {}
                radar_plot[attr]['success']=success;radar_plot[attr]['precision']=precision
                radar_plot[attr]['EIoU']=EIoU;radar_plot[attr]['num'] = len(videos)
            draw_radar_plot(radar_plot,trackers,args.dataset)

    elif 'SV248S' in args.dataset:
        dataset = SV248SDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        EIoU_success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_EIou,
                trackers), desc='eval EIoU', total=len(trackers), ncols=100):
                EIoU_success_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,EIoU_success_ret,
                show_video_level=args.show_video_level)
        #是否绘制跟踪结果
        if args.vis:
            radar_plot = {}
            for attr, videos in dataset.attr.items():
                success,precision,EIoU=draw_success_precision(success_ret,
                            name=dataset.name,
                            videos=videos,
                            attr=attr,
                            bold_name='SiamDTF(Ours)',
                            precision_ret=precision_ret,
                            EIoU_success_ret = EIoU_success_ret,
                            dataset_name=args.dataset)
                radar_plot[attr] = {}
                radar_plot[attr]['success']=success;radar_plot[attr]['precision']=precision
                radar_plot[attr]['EIoU']=EIoU;radar_plot[attr]['num'] = len(videos)
            draw_radar_plot(radar_plot,trackers,args.dataset)
                
    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        # success_ret = benchmark.eval_success(trackers)
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            draw_success_precision(success_ret,
                    name=dataset.name,
                    videos=dataset.attr['ALL'],
                    attr='ALL',
                    precision_ret=precision_ret,
                    norm_precision_ret=norm_precision_ret)
    elif 'UAV' in args.dataset:
        dataset = UAVDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                        name=dataset.name,
                        videos=videos,
                        attr=attr,
                        precision_ret=precision_ret)
    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
        if args.vis:
            for attr, videos in dataset.attr.items():
                draw_success_precision(success_ret,
                            name=dataset.name,
                            video=videos,
                            attr=attr,
                            precision_ret=precision_ret)
    elif 'VOT2018' == args.dataset or 'VOT2016' == args.dataset:
        dataset = VOTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)
        # benchmark.show_result(ar_result)

        benchmark = EAOBenchmark(dataset)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        # benchmark.show_result(eao_result)
        ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)
    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(tracker_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=args.show_video_level)
        if args.vis:
            draw_f1(f1_result)

