import matplotlib.pyplot as plt
import numpy as np
import os
from .draw_utils import COLOR, LINE_STYLE

def draw_success_precision(success_ret, name, videos, attr, dataset_name,precision_ret=None,EIoU_success_ret=None,
        norm_precision_ret=None, bold_name=None, axis=[0, 1]):
    # success plot
    file_path = os.path.join(os.path.dirname(__file__),'..','..','..')
    sv_path = os.path.join(file_path,dataset_name +'_ablation_fig_20240404')
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    fig, ax = plt.subplots()
    ax.grid(visible=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots of OPE on %s}' % (name))
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                color=COLOR[tracker_name], linestyle=LINE_STYLE[tracker_name],label=label, linewidth=2)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    ymin = 0
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    plt.tight_layout()
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    if attr == 'ALL':
        plt.savefig(os.path.join(sv_path, name + '_succ.png'))
    else:
        plt.savefig(os.path.join(sv_path, attr + '_succ.png'))
    plt.show()

    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(visible=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[tracker_name], linestyle=LINE_STYLE[tracker_name],label=label, linewidth=2)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        plt.tight_layout()
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        if attr == 'ALL':
            plt.savefig(os.path.join(sv_path, name + '_pre.png'))
        else:
            plt.savefig(os.path.join(sv_path, attr + '_pre.png'))
        plt.show()

    if EIoU_success_ret:
        fig, ax = plt.subplots()
        ax.grid(visible=True)
        ax.set_aspect(1)
        plt.xlabel('EIoU threshold')
        plt.ylabel('Success rate')
        if attr == 'ALL':
            plt.title(r'\textbf{EIou Success plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{EIoU Success plots of OPE - %s}' % (attr))
        plt.axis([0, 1]+axis)
        EIoU = {}
        thresholds = np.arange(0, 1.05, 0.05)
        for tracker_name in EIoU_success_ret.keys():
            value = [v for k, v in EIoU_success_ret[tracker_name].items() if k in videos]
            EIoU[tracker_name] = np.mean(value)
        for idx, (tracker_name, auc) in  \
                enumerate(sorted(EIoU.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
            else:
                label = "[%.3f] " % (auc) + tracker_name
            value = [v for k, v in EIoU_success_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[tracker_name], linestyle=LINE_STYLE[tracker_name],label=label, linewidth=2)
        # ax.legend(loc='lower left', labelspacing=0.2)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        plt.tight_layout()
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        if attr == 'ALL':
            plt.savefig(os.path.join(sv_path, name + '_EIoU_succ.png'))
        else:
            plt.savefig(os.path.join(sv_path, attr + '_EIoU_succ.png'))
        plt.close()
    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        ax.grid(visible=True)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[tracker_name], linestyle=LINE_STYLE[tracker_name],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        ymin = 0
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
    return success,precision,EIoU
    
def draw_radar_plot(data:dict,tracker_names,dataset_name):
    file_path = os.path.join(os.path.dirname(__file__),'..','..','..')
    sv_path = os.path.join(file_path,dataset_name + '_radar map_20240404')
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    lable_name = list(data.keys())    # data: attr:precision:tracker:score                                  
    score = {}  
    if 'precision' in list(data[lable_name[0]].keys()):
        for id,tracker in enumerate(tracker_names):
            score[tracker] = {}
            score[tracker]['precision'] = []
            for attr,value in data.items():
                number = np.array(list((value['precision']).values()))
                if np.max(number) == np.min(number):
                    times = 1
                else:
                    times = 1 / (np.max(number)-np.min(number))
                precision_value_1 = np.exp(times * number)
                precision_value = precision_value_1 / np.max(precision_value_1)
                index = np.where(tracker == np.array(list(value['precision'].keys())))
                score[tracker]['precision'].append(precision_value[index])
        dim_num = len(data.items())
        radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
        radians = np.concatenate((radians, [radians[0]]))
        fig, ax = plt.subplots(figsize=(7,5),subplot_kw=dict(polar=True))
        plt.title(r'\textbf{Attributes of Precision Radar Map on %s}' % (dataset_name))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['polar'].set_visible(False)
        for tracker in tracker_names:
            score_precision = np.array(score[tracker]['precision'])
            score_precision = np.concatenate((score_precision,[score_precision[0]]))

            max_indexs = np.where(score_precision[:-1]==1)[0]
            for max_index in max_indexs:
                text_value = data[lable_name[max_index]]['precision'][tracker] * 100
                text_attr_num = data[lable_name[max_index]]['num']
                rotation_attr_angle = rotation(radians[max_index])
                rotation_value_agnle = rotation(radians[max_index]+0.15)
                ax.text(radians[max_index]+0.15, score_precision[max_index]+0.1, f'{text_value:.1f}', 
                color=COLOR[tracker], weight='bold',ha='center',va='center',clip_on= False,rotation=rotation_value_agnle)
                if text_attr_num == data['ALL']['num']:
                    ax.text(radians[max_index]-0.15, score_precision[max_index]+0.1, '({:d})'.format(text_attr_num), 
                            color='black', weight='bold',ha='center',va='center',clip_on= False)
                else:
                    ax.text(radians[max_index], score_precision[max_index]+0.22, '({:d})'.format(text_attr_num), 
                            color='black', weight='bold',ha='center',va='center',clip_on= False,rotation=rotation_attr_angle)

            min_value = 1 / np.exp(1)
            min_indexs = np.where(abs(min_value-score_precision[:-1])<1/np.exp(10))[0]
            for min_index in min_indexs:
                text_value = data[lable_name[min_index]]['precision'][tracker] * 100
                ax.text(radians[min_index], score_precision[min_index]-0.1, f'{text_value:.1f}', 
                color=COLOR[tracker], weight='bold',ha='center',va='center',clip_on= False,)

            ax.plot(radians,score_precision, marker='o',markersize=6,color=COLOR[tracker],clip_on=False)
        lable = np.concatenate((lable_name,[lable_name[0]]))
        ax.set_thetagrids(radians*180/np.pi, lable)                  
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 1)
        ax.set_yticklabels([])
        # ax.set_rlabel_position(285)
        plt.legend(tracker_names,bbox_to_anchor=(1.05, 1.03), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(os.path.join(sv_path,'precision.tiff'),dpi=500)
        plt.close()

    if 'success' in list(data[lable_name[0]].keys()):
        for id,tracker in enumerate(tracker_names):
            score[tracker]['success'] = []
            for attr,value in data.items():
                number = np.array(list((value['success']).values()))
                if np.max(number) == np.min(number):
                    times = 1
                else:
                    times = 1 / (np.max(number)-np.min(number))
                success_value1 = np.exp(times * number)
                success_value = success_value1 / np.max(success_value1)
                index = np.where(tracker == np.array(list(value['success'].keys())))
                score[tracker]['success'].append(success_value[index])
        dim_num = len(data.items())
        radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
        radians = np.concatenate((radians, [radians[0]]))
        fig, ax = plt.subplots(figsize=(7,5),subplot_kw=dict(polar=True))
        plt.title(r'\textbf{Attributes of Success Radar Map on %s}' % (dataset_name))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['polar'].set_visible(False)

        for tracker in tracker_names:
            score_success = np.array(score[tracker]['success'])
            score_success = np.concatenate((score_success,[score_success[0]]))

            max_indexs = np.where(score_success[:-1]==1)[0]
            for max_index in max_indexs:
                text_value = data[lable_name[max_index]]['success'][tracker] * 100
                text_attr_num = data[lable_name[max_index]]['num']
                rotation_attr_angle = rotation(radians[max_index])
                rotation_value_agnle = rotation(radians[max_index]+0.15)
                ax.text(radians[max_index]+0.15, score_success[max_index]+0.1, f'{text_value:.1f}', 
                color=COLOR[tracker], weight='bold',ha='center',va='center',clip_on= False,rotation=rotation_value_agnle)
                if text_attr_num == data['ALL']['num']:
                    ax.text(radians[max_index]-0.15, score_success[max_index]+0.1, '({:d})'.format(text_attr_num), 
                            color='black', weight='bold',ha='center',va='center',clip_on= False)
                else:
                    ax.text(radians[max_index], score_success[max_index]+0.22, '({:d})'.format(text_attr_num), 
                            color='black', weight='bold',ha='center',va='center',clip_on= False,rotation=rotation_attr_angle)

            min_value = 1 / np.exp(1)
            min_indexs = np.where(abs(min_value-score_success[:-1])<1/np.exp(10))[0]
            for min_index in min_indexs:
                text_value = data[lable_name[min_index]]['success'][tracker] * 100
                ax.text(radians[min_index], score_success[min_index]-0.1, f'{text_value:.1f}', 
                color=COLOR[tracker], weight='bold',ha='center',va='center',clip_on= False,)

            ax.plot(radians,score_success, marker='o',markersize=6,color=COLOR[tracker],clip_on=False)

        lable = np.concatenate((lable_name,[lable_name[0]]))
        ax.set_thetagrids(radians*180/np.pi, lable)                  
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 1)
        ax.set_yticklabels([])
        # ax.set_rlabel_position(285)
        plt.legend(tracker_names,bbox_to_anchor=(1.05, 1.03), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(os.path.join(sv_path,'success.tiff'),dpi=500)
        plt.close()

    if 'EIoU' in list(data[lable_name[0]].keys()):
        for id,tracker in enumerate(tracker_names):
            score[tracker]['EIoU'] = []
            for attr,value in data.items():       
                number = np.array(list((value['EIoU']).values()))
                if np.max(number) == np.min(number):
                    times = 1
                else:
                    times = 1 / (np.max(number)-np.min(number))
                EIoU_success_value1 = np.exp(times * number)
                EIoU_success_value = EIoU_success_value1 / np.max(EIoU_success_value1)
                index = np.where(tracker == np.array(list(value['EIoU'].keys())))
                score[tracker]['EIoU'].append(EIoU_success_value[index])
        dim_num = len(data.items())
        radians = np.linspace(0, 2 * np.pi, dim_num, endpoint=False)
        radians = np.concatenate((radians, [radians[0]]))
        fig, ax = plt.subplots(figsize=(7,5),subplot_kw=dict(polar=True))
        plt.title(r'\textbf{Attributes of EIoU Radar Map on %s}' % (dataset_name))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.spines['polar'].set_visible(False)

        for tracker in tracker_names:
            score_success = np.array(score[tracker]['EIoU'])
            score_success = np.concatenate((score_success,[score_success[0]]))

            max_indexs = np.where(score_success[:-1]==1)[0]
            for max_index in max_indexs:
                text_value = data[lable_name[max_index]]['EIoU'][tracker] * 100
                text_attr_num = data[lable_name[max_index]]['num']
                rotation_attr_angle = rotation(radians[max_index])
                rotation_value_agnle = rotation(radians[max_index]+0.15)
                ax.text(radians[max_index]+0.15, score_success[max_index]+0.1, f'{text_value:.1f}', 
                color=COLOR[tracker], weight='bold',ha='center',va='center',clip_on= False,rotation=rotation_value_agnle)
                if text_attr_num == data['ALL']['num']:
                    ax.text(radians[max_index]-0.15, score_success[max_index]+0.1, '({:d})'.format(text_attr_num), 
                            color='black', weight='bold',ha='center',va='center',clip_on= False)
                else:
                    ax.text(radians[max_index], score_success[max_index]+0.22, '({:d})'.format(text_attr_num), 
                            color='black', weight='bold',ha='center',va='center',clip_on= False,rotation=rotation_attr_angle)
            
            min_value = 1 / np.exp(1)
            min_indexs = np.where(abs(min_value-score_success[:-1])<1/np.exp(10))[0]
            for min_index in min_indexs:
                text_value = data[lable_name[min_index]]['EIoU'][tracker] * 100
                ax.text(radians[min_index], score_success[min_index]-0.1, f'{text_value:.1f}', 
                color=COLOR[tracker], weight='bold',ha='center',va='center',clip_on= False,)

            ax.plot(radians,score_success, marker='o',markersize=6,color=COLOR[tracker],clip_on=False)

        lable = np.concatenate((lable_name,[lable_name[0]]))
        ax.set_thetagrids(radians*180/np.pi, lable)                  
        ax.set_theta_zero_location('N')
        ax.set_rlim(0, 1)
        ax.set_yticklabels([])
        # ax.set_rlabel_position(285)
        plt.legend(tracker_names,bbox_to_anchor=(1.05, 1.03), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        plt.savefig(os.path.join(sv_path,'EIoU.tiff'),dpi=500)
        plt.close()

def rotation(x):
    pi = np.pi
    if x < pi / 2:
        return (x/pi) * 180
    elif x < pi:
        return -(pi-x) * 180 / pi
    elif x < pi * 1.5:
        return (x-pi) * 180 / pi
    elif x < pi * 2:
        return -(2*pi - x) * 180 / pi
