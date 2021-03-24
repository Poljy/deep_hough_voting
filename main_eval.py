import os
import sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
sys.path.append(os.path.join(ROOT_DIR, 'losses'))

from votenet import VoteNet
from votenet_loss import votenet_loss
from ap_helper import APCalculator, parse_predictions, parse_groundtruths
from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
from model_util_sunrgbd import SunrgbdDatasetConfig
from dump_helper import dump_results

###### CONFIGURATION ######

#Evaluation parameters
BATCH_SIZE = 8
NUM_POINT = 10000
AP_IOU_THRESHOLDS = [0.25,0.5]
conf_thresh = 0.05
nms_iou = 0.25
use_color = False
use_height = True

#Storing parameters
LOG_DIR = 'log_train'
CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')

DUMP_DIR = os.path.join(LOG_DIR, 'evaluation_results')

## PREPARING DUMP DIRECTORY

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
def log_string(out_str):
    DUMP_FOUT.write(out_str+'\n')
    DUMP_FOUT.flush()
    print(out_str)

## INITIALIZE DATASET

DATASET_CONFIG = SunrgbdDatasetConfig()
TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT, augment=False, use_color=use_color, use_height=use_height, use_v1=True)
print(len(TEST_DATASET))

## INITIALIZE DATALOADER

TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(len(TEST_DATALOADER))

## INITIALIZE MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NC = DATASET_CONFIG.num_class
NH = DATASET_CONFIG.num_heading_bin
NS = DATASET_CONFIG.num_size_cluster
mean_sizes = DATASET_CONFIG.mean_size_arr

pc_features = 0
if use_height:
    pc_features+=1
if use_color:
    pc_features+=3

features_dim = 256
n_proposals = 256

net = VoteNet(NC=NC, NH=NH, NS=NS, mean_sizes=mean_sizes, pc_features=pc_features, features_dim=features_dim, n_proposals=n_proposals)
net.to(device)

## INITIALIZE_LOSS AND OPTIMIZER

criterion = votenet_loss
optimizer = optim.Adam(net.parameters(), lr=0.001)

## LOAD CHECKPOINT

if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': True, 'use_3d_nms': False, 'nms_iou': nms_iou,
    'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
    'conf_thresh': conf_thresh, 'dataset_config':DATASET_CONFIG}

###### EVALUATION ######

def evaluate_one_epoch():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
        for iou_thresh in AP_IOU_THRESHOLDS]
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = net(inputs)

        # Compute loss
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT) 
        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT) 
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
    
        # Dump evaluation results for visualization
        if batch_idx == 0:
            dump_results(end_points, DUMP_DIR, DATASET_CONFIG)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-'*10, 'iou_thresh: %f'%(AP_IOU_THRESHOLDS[i]), '-'*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()

if __name__=='__main__':
    eval()
