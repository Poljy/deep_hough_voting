import os
import sys
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
sys.path.append(os.path.join(ROOT_DIR, 'losses'))

from votenet import VoteNet
from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset
from model_util_sunrgbd import SunrgbdDatasetConfig
from votenet_loss import votenet_loss
from pytorch_utils import BNMomentumScheduler
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

###### CONFIGURATION ######

#Learning and evaluation parameters
BATCH_SIZE = 8
NUM_POINT = 10000
MAX_EPOCH = 180
use_color = False
use_height = True
BASE_LEARNING_RATE = 0.001
BN_DECAY_STEP = 20
BN_DECAY_RATE = 0.5
LR_DECAY_STEPS = [80,120,160]
LR_DECAY_RATES = [0.1,0.1,0.1]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
ap_iou_threshold = 0.25

#Storing parameters
LOG_DIR = 'log_train'
DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')

## PREPARING LOG DIRECTORY

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)


## INITIALIZE DATASET

DATASET_CONFIG = SunrgbdDatasetConfig()
TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT, augment=True, use_color=use_color, use_height=use_height, use_v1=True)
TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT, augment=False, use_color=use_color, use_height=use_height, use_v1=True)
        
print(len(TRAIN_DATASET), len(TEST_DATASET))

## INITIALIZE DATALOADER

TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

## INITIALIZE MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pc_features = 3
if use_height:
    pc_features+=1
if use_color:
    pc_features+=3

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
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE)

## LOAD CHECKPOINT

it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

## INITIALIZE BATCH NORM AND LR SCHEDULES

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': True, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

###### TRAINING ######

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        end_points = net(inputs)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label:
            assert(key not in end_points)
            end_points[key] = batch_data_label[key]
        loss, end_points = criterion(end_points, DATASET_CONFIG)
        loss.backward()
        optimizer.step()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    ap_calculator = APCalculator(ap_iou_thresh=ap_iou_threshold,
        class2type_map=DATASET_CONFIG.class2type)
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
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Log statistics
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    # Evaluate average precision
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        log_string('eval %s: %f'%(key, metrics_dict[key]))

    mean_loss = stat_dict['loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
