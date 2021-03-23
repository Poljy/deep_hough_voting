import torch
import torch.nn as nn
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance

def compute_objectness_loss(dict_output):

    # Parameters
    thresh1 = 0.6
    thresh0 = 0.3
    weighted_cross_entropy = nn.CrossEntropyLoss(torch.Tensor([0.2,0.8]).cuda(), reduction='none')

    # Retrieve ground truth centers, predicted centers (i.e. aggregated votes) and the objectness scores
    predicted_centers = dict_output['aggregated_vote_xyz']
    gt_center = dict_output['center_label'][:,:,0:3]
    objectness_scores = dict_output['objectness_scores']

    # Retrieve useful dimensions
    batch_size,n_pred_centers,n_gt_centers = gt_center.shape[0],predicted_centers.shape[1],gt_center.shape[1]

    # Compute nearest neighbour distances (and correspondances) between predicted and ground truth centers
    dist1, object_assignment, _ , _ = nn_distance(predicted_centers, gt_center)
    dist1 = torch.sqrt(dist1+1e-6)

    # Compute objectness label: we consider that a predicted center belongs to an object if its nn distance is below thresh0
    objectness_label = torch.zeros((batch_size,n_pred_centers), dtype=torch.long).cuda()
    objectness_label[dist1<thresh0] = 1

    # Compute objectness mask: if the nn distance is in between the two thresholds, we ignore this prediction
    objectness_mask = torch.zeros((batch_size,n_pred_centers)).cuda()
    objectness_mask[dist1<thresh0] = 1
    objectness_mask[dist1>thresh1] = 1    

    # Compute objectness losses and average over the unmasked centers
    objectness_loss = weighted_cross_entropy(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    return objectness_loss, objectness_label, objectness_mask, object_assignment