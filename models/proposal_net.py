import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

import pointnet2_utils
from pointnet2_modules import PointnetSAModuleVotes

def decode_scores(net, dict_output, NC, NH, NS, mean_sizes):

    net_transposed = net.transpose(2,1)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    #Parse output
    objectness_scores = net_transposed[:,:,0:2]
    center_offsets = net_transposed[:,:,2:5]
    heading_scores = net_transposed[:,:,5:5+NH]
    heading_residuals_normalized = net_transposed[:,:,5+NH:5+NH*2]
    size_scores = net_transposed[:,:,5+NH*2:5+NH*2+NS]
    size_residuals_normalized = net_transposed[:,:,5+NH*2+NS:5+NH*2+NS*4].view([batch_size, num_proposal, NS, 3]) 
    sem_cls_scores = net_transposed[:,:,5+NH*2+NS*4:]

    #Objectness classification scores
    dict_output['objectness_scores'] = objectness_scores
    
    #Center regression offset
    dict_output['center'] = dict_output['aggregated_vote_xyz'] + center_offsets

    #Classification scores for heading bins (i.e. bounding box orientations)
    dict_output['heading_scores'] = heading_scores

    #Regression offsets for heading bins (i.e. bounding box orientations)
    dict_output['heading_residuals_normalized'] = heading_residuals_normalized 
    dict_output['heading_residuals'] = heading_residuals_normalized * (np.pi/NH)

    #Classification scores for size templates
    dict_output['size_scores'] = size_scores

    #Scale regression offsets for size templates (height, width and length)
    dict_output['size_residuals_normalized'] = size_residuals_normalized
    dict_output['size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_sizes.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    #Semantic classification scores
    dict_output['sem_cls_scores'] = sem_cls_scores

    return dict_output

class ProposalNet(nn.Module):

    def __init__(self, in_features=256, n_proposals=128, NC=10, NH=12, NS=10, mean_sizes=None):

        super().__init__()

        #Parameters of the network: dim of input features, number of box proposals, number of classes, number of heading bins, number of size templates and corresponding mean sizes
        self.in_features = in_features
        self.n_proposals = n_proposals
        self.NC = NC
        self.NH = NH
        self.NS = NS
        if mean_sizes is None:
            self.mean_sizes = np.array([[0.765840, 1.398258, 0.472728], [2.114256, 1.620300, 0.927272], [0.404671, 1.071108, 1.688889], [0.591958, 0.552978, 0.827272], [0.695190, 1.346299, 0.736364], [0.528526, 1.002642, 1.172878], [0.500618, 0.632163, 0.683424], [0.923508, 1.867419, 0.845495], [0.791118, 1.279516, 0.718182], [0.699104, 0.454178, 0.756250]])
        else:
          self.mean_sizes = mean_sizes

        #Layers
        self.sa = PointnetSAModuleVotes(npoint=self.n_proposals,radius=0.3,nsample=16, mlp=[self.in_features, 128, 128, 128], use_xyz=True, normalize_xyz=True)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128,5+2*self.NH+4*self.NS+self.NC,1)

    def forward(self, xyz, features, dict_output):
        
        #Cluster votes
        xyz, features, fps_inds = self.sa(xyz, features)
        dict_output['aggregated_vote_xyz'] = xyz 
        dict_output['aggregated_vote_inds'] = fps_inds 

        #Generate proposals
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net)

        #Decode scores
        dict_output = decode_scores(net, dict_output, self.NC, self.NH, self.NS, self.mean_sizes)

        return dict_output