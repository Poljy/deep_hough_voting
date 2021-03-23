import torch
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from nn_distance import nn_distance

def compute_vote_loss(dict_output):

    max_gt_votes = 3 #Maximum number of ground truth votes that a point can store (see appendix A.1 to learn about this parameter)

    #Retrieve useful dimensions
    batch_size,n_seeds = dict_output['seed_xyz'].shape[0],dict_output['seed_xyz'].shape[1]

    #Retrieve predicted votes and their corresponding seeds (network output)
    vote_xyz = dict_output['vote_xyz']
    seed_inds = dict_output['seed_inds'].long()

    #Retrieve the ground truth vote mask for the seeds whether (i.e. do the seeds have a vote mask)
    seed_gt_votes_mask = torch.gather(dict_output['vote_label_mask'], 1, seed_inds)

    #Retrieve the ground truth votes for the seeds
    seed_inds_expand = seed_inds.view(batch_size,n_seeds,1).repeat(1,1,3*max_gt_votes)
    seed_gt_votes = torch.gather(dict_output['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += dict_output['seed_xyz'].repeat(1,1,3)

    #Reshape the votes and ground truth votes to compute the minimum vote distance
    vote_xyz_reshape = vote_xyz.view(batch_size*n_seeds, 1, 3) 
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*n_seeds, max_gt_votes, 3)

    #Compute the nn_distance (i.e. distance with nearest neighbors) and retrieve the minimum one
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1)

    #Consider only significant votes using the mask, and retrieve the vote loss as the average distance between predicted votes and ground truth
    votes_dist = votes_dist.view(batch_size, n_seeds)
    loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)

    return loss