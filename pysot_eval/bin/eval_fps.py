import argparse
import numpy as np
from multiprocessing import Pool
import os
import tqdm

def calculate_fps(trackers,result_path):
    fps_total = {}

    for tracker in trackers:
        time = []
        files = os.listdir(os.path.join(result_path,tracker))[:65]
        for file in files:
            result_file = os.path.join(result_path,tracker,file,file + '_time.txt')
            times = np.loadtxt(result_file,dtype=np.float32)
            time += list(times)
        fps = len(time) / np.sum(time)
        fps_total[tracker] = fps
        
    return fps_total

if __name__=='__main__':
    result_path = '/home/zhoujiawei/tracking_model/SiamCAR-master/results/SatSOT'   # SatSOT SV248S
    trackers = ['SiamDFT(Ours)']
    num = len(trackers)
    parser = argparse.ArgumentParser(description='Single Object Tracking Evaluation')
    parser.add_argument('--tracker_result_dir', type=str, default=result_path, help='tracker result root')#tracker 结果路径
    parser.add_argument('--trackers', default=trackers, nargs='+',help='Trackers name')#nargs='+' 设置一个或多个参数， 同时评估多个tracker
    parser.add_argument('--num', type=int, help='number of processes to eval', default=num)#线程数目，有默认值 1，不是很重要
    args = parser.parse_args()
    fps = {}
    # with Pool(processes=args.num) as pool:
    #     for ret in tqdm(pool.imap_unordered(calculate_fps,
    #         args.trackers,args.tracker_result_dir), desc='calculate fps', total=len(trackers), ncols=100):
    #         fps.update(ret)
    fps.update(calculate_fps(args.trackers,args.tracker_result_dir))
    save_path_file = os.path.join(result_path,'_time.txt')
    for key,value in fps.items():
        with open (save_path_file,'+a') as f:
              f.write('{:18s}:{:0.2f} \n'.format(key,value))
        print('{:18s}:{:0.2f} \n'.format(key,value))