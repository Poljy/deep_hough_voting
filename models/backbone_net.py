import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule

class BackboneNet(nn.Module):

    def __init__(self,in_features):
        super().__init__()

        ## Architecture details come from Section 4.3 / Appendix A of the paper + Table 5 + Official Implementation

        # Set abstraction layers
        self.sa1 = PointnetSAModuleVotes(mlp=[in_features,64,64,128],radius=0.2,npoint=2048,nsample=64,use_xyz=True,normalize_xyz=True)
        self.sa2 = PointnetSAModuleVotes(mlp=[128, 128, 128, 256],radius=0.4,npoint=1024,nsample=32,use_xyz=True,normalize_xyz=True)
        self.sa3 = PointnetSAModuleVotes(mlp=[256, 128, 128, 256],radius=0.8,npoint=512,nsample=16,use_xyz=True,normalize_xyz=True)
        self.sa4 = PointnetSAModuleVotes(mlp=[256, 128, 128, 256],radius=1.2,npoint=256,nsample=16,use_xyz=True,normalize_xyz=True)

        #Feature propagation/upsampling layers
        self.fp1 = PointnetFPModule(mlp=[512,256,256])
        self.fp2 = PointnetFPModule(mlp=[512,256,256])

    def forward(self,pc):

        ##Decompose points as xyz coordinates + features
        xyz = pc[:,:,:3].contiguous()
        if pc.shape[2] > 3:
            features = pc[:,:,3:].transpose(1, 2).contiguous() #We transpose because it is the right format for the following layers
        else:
            features = None
        
        ##Apply SA layers and store their successive outputs
        dict_output = {}

        #1
        xyz, features, fps_inds = self.sa1(xyz, features)
        dict_output['sa1_xyz'],dict_output['sa1_features'],dict_output['sa1_inds']  = xyz, features, fps_inds

        #2
        xyz, features, fps_inds = self.sa2(xyz, features)
        dict_output['sa2_xyz'],dict_output['sa2_features'],dict_output['sa2_inds']  = xyz, features, fps_inds

        #3
        xyz, features, fps_inds = self.sa3(xyz, features)
        dict_output['sa3_xyz'],dict_output['sa3_features'],dict_output['sa3_inds']  = xyz, features, fps_inds

        #4
        xyz, features, fps_inds = self.sa4(xyz, features)
        dict_output['sa4_xyz'],dict_output['sa4_features'],dict_output['sa4_inds']  = xyz, features, fps_inds

        ##Apply FP layers using the outputs of previous SA layers

        #1
        features = self.fp1(dict_output['sa3_xyz'], dict_output['sa4_xyz'], dict_output['sa3_features'], dict_output['sa4_features'])

        #2
        features = self.fp2(dict_output['sa2_xyz'], dict_output['sa3_xyz'], dict_output['sa2_features'], features)

        dict_output['fp2_xyz'] = dict_output['sa2_xyz']
        dict_output['fp2_features'] = features
        dict_output['fp2_inds'] = dict_output['sa1_inds'][:,0:dict_output['fp2_xyz'].shape[1]]

        return dict_output
