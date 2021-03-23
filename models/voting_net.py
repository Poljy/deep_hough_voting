import torch
import torch.nn as nn
import torch.nn.functional as F

class VotingNet(nn.Module):

    def __init__(self, features_dim):

        super().__init__()

        #Number of features per seed/vote
        self.features_dim = features_dim

        #Layers of the voting module
        self.conv1 = torch.nn.Conv1d(self.features_dim, self.features_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.features_dim, self.features_dim, 1)
        self.conv3 = torch.nn.Conv1d(self.features_dim, 3+self.features_dim, 1)
        self.bn1 = torch.nn.BatchNorm1d(self.features_dim)
        self.bn2 = torch.nn.BatchNorm1d(self.features_dim)
        
    def forward(self, seed_xyz, seed_features):

        #Retrieve batch size and number of seeds
        batch_size = seed_xyz.shape[0]
        n_seeds = seed_xyz.shape[1]

        #Apply shared MLP to the seed features
        net = F.relu(self.bn1(self.conv1(seed_features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) 
        
        #Retrieve xyz offset and features offset from the output
        net = net.transpose(2,1).view(batch_size, n_seeds, 3+self.features_dim)
        xyz_offset,features_offset = net[:,:,0:3],net[:,:,3:]

        #Vote_xyz = Seed_xyz + delta(Vote_xyz) (see section 4.1 of the paper)
        vote_xyz = seed_xyz + xyz_offset
        vote_xyz = vote_xyz.contiguous().view(batch_size, n_seeds, 3)
        
        #Vote_features = Seed_features + delta(Vote_features)
        vote_features = seed_features.transpose(2,1) + features_offset
        vote_features = vote_features.contiguous().view(batch_size, n_seeds, self.features_dim)
        vote_features = vote_features.transpose(2,1).contiguous()
        
        return vote_xyz, vote_features
