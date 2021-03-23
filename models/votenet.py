import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_net import BackboneNet
from voting_net import VotingNet
from proposal_net import ProposalNet


class VoteNet(nn.Module):

    def __init__(self, NC, NH, NS, mean_sizes, pc_features=0, features_dim=256, n_proposals=128):
        super().__init__()

        #Dimension parameters
        self.NC = NC
        self.NH = NH
        self.NS = NS
        self.mean_sizes = mean_sizes
        self.pc_features = pc_features
        self.features_dim = features_dim
        self.n_proposals = n_proposals

        # Backbone Network
        self.backbone_net = BackboneNet(in_features=self.pc_features)

        # Voting Network
        self.vgen = VotingNet(features_dim=self.features_dim)

        # Proposal Network
        self.pnet = ProposalNet(in_features=self.features_dim,n_proposals=n_proposals,NC=self.NC,NH=self.NH,NS=self.NS,mean_sizes=self.mean_sizes)

    def forward(self, inputs):

        #Apply backbone network
        dict_output = self.backbone_net(inputs['point_clouds'])
                
        #Retrive seeds coordinates, features and indices
        xyz = dict_output['fp2_xyz']
        features = dict_output['fp2_features']
        dict_output['seed_inds'] = dict_output['fp2_inds']
        dict_output['seed_xyz'] = xyz
        dict_output['seed_features'] = features
        
        #Apply voting network
        xyz, features = self.vgen(xyz, features)

        #Retrieve votes coordinates and normalized features
        features_norm = torch.norm(features, p=2, dim=1,keepdim=True)
        features = features.div(features_norm)
        dict_output['vote_xyz'] = xyz
        dict_output['vote_features'] = features

        #Apply proposal network
        dict_output = self.pnet(xyz, features, dict_output)

        return dict_output

