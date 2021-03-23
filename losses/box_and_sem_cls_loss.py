import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance, huber_loss

def compute_center_loss(pred_center,gt_center,objectness_label,box_label_mask):

    ## Compute center loss: Chamfer loss as explained in appendix A.1
    dist1, _ , dist2, _ = nn_distance(pred_center, gt_center)
    centroid_reg_loss1 = torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    return center_loss

def compute_heading_loss(heading_class_label,heading_residual_label,heading_scores, heading_residual_normalized, object_assignment,objectness_label,num_heading_bin):

    batch_size = object_assignment.shape[0]

    ## Compute heading (angle) loss: bin classification 
    heading_class_label = torch.gather(heading_class_label, 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(heading_scores.transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    ## Compute heading (angle) loss: residual regression (i.e. distance from its bin center)
    heading_residual_label = torch.gather(heading_residual_label, 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_() # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(heading_residual_normalized*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    return heading_class_loss,heading_residual_normalized_loss

def compute_size_loss(size_class_label,size_residual_label,size_scores,size_residuals_normalized, object_assignment,objectness_label, num_size_cluster, mean_size_arr):

    batch_size = object_assignment.shape[0]

    ## Compute size loss: classification
    size_class_label = torch.gather(size_class_label, 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(size_scores.transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    ## Compute size loss: residual regression
    size_residual_label = torch.gather(size_residual_label, 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(size_residuals_normalized*size_label_one_hot_tiled, 2) # (B,K,3)
    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    return size_class_loss,size_residual_normalized_loss

def compute_semantic_classification_loss(sem_cls_label,sem_cls_scores,object_assignment,objectness_label):
    
    sem_cls_label = torch.gather(sem_cls_label, 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(sem_cls_scores.transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return sem_cls_loss


def compute_box_and_sem_cls_loss(dict_output, config):

    ## Retrieve config
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    ## Retrieve useful elements
    object_assignment = dict_output['object_assignment']
    pred_center = dict_output['center']
    gt_center = dict_output['center_label'][:,:,0:3]
    box_label_mask = dict_output['box_label_mask']
    objectness_label = dict_output['objectness_label'].float()

    heading_class_label = dict_output['heading_class_label']
    heading_scores = dict_output['heading_scores']
    heading_residual_label = dict_output['heading_residual_label']
    heading_residual_normalized = dict_output['heading_residuals_normalized']

    size_class_label = dict_output['size_class_label']
    size_scores = dict_output['size_scores']
    size_residual_label = dict_output['size_residual_label']
    size_residuals_normalized = dict_output['size_residuals_normalized']

    sem_cls_label = dict_output['sem_cls_label']
    sem_cls_scores = dict_output['sem_cls_scores']

    ## Compute center loss
    center_loss = compute_center_loss(pred_center,gt_center,objectness_label,box_label_mask)

    ## Compute heading loss
    heading_class_loss,heading_residual_normalized_loss = compute_heading_loss(heading_class_label,heading_residual_label,heading_scores, heading_residual_normalized, object_assignment,objectness_label,num_heading_bin)

    ## Compute size loss
    size_class_loss,size_residual_normalized_loss = compute_size_loss(size_class_label,size_residual_label,size_scores,size_residuals_normalized, object_assignment,objectness_label, num_size_cluster, mean_size_arr)

    ## Compute semantic classification loss
    sem_cls_loss = compute_semantic_classification_loss(sem_cls_label,sem_cls_scores,object_assignment,objectness_label)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss