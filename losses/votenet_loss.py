import torch
from vote_loss import compute_vote_loss
from objectness_loss import compute_objectness_loss
from box_and_sem_cls_loss import compute_box_and_sem_cls_loss


def votenet_loss(dict_output, config, loss_weights=None):

    if loss_weights is None:
        loss_weights = {}
        loss_weights['center_loss'] = 1.0
        loss_weights['heading_cls_loss'] = 0.1
        loss_weights['heading_reg_loss'] = 1.0
        loss_weights['size_cls_loss'] = 0.1
        loss_weights['size_reg_loss'] = 0.1
        loss_weights['vote_loss'] = 1.0
        loss_weights['objectness_loss'] = 0.5
        loss_weights['box_loss'] = 1.0
        loss_weights['sem_cls_loss'] = 0.1

    # Vote loss
    vote_loss = compute_vote_loss(dict_output)
    dict_output['vote_loss'] = vote_loss

    # Objectness loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(dict_output)
    dict_output['objectness_loss'] = objectness_loss
    dict_output['objectness_label'] = objectness_label
    dict_output['objectness_mask'] = objectness_mask
    dict_output['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    dict_output['pos_ratio'] = torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    dict_output['neg_ratio'] = torch.sum(objectness_mask.float())/float(total_num_proposal) - dict_output['pos_ratio']

    # Box loss and semantic classification loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = compute_box_and_sem_cls_loss(dict_output, config)
    dict_output['center_loss'] = center_loss
    dict_output['heading_cls_loss'] = heading_cls_loss
    dict_output['heading_reg_loss'] = heading_reg_loss
    dict_output['size_cls_loss'] = size_cls_loss
    dict_output['size_reg_loss'] = size_reg_loss
    dict_output['sem_cls_loss'] = sem_cls_loss
    box_loss = loss_weights['center_loss']*center_loss + loss_weights['heading_cls_loss']*heading_cls_loss + loss_weights['heading_reg_loss']*heading_reg_loss + loss_weights['size_cls_loss']*size_cls_loss + loss_weights['size_reg_loss']*size_reg_loss
    dict_output['box_loss'] = box_loss

    # Final loss function
    loss = loss_weights['vote_loss']*vote_loss + loss_weights['objectness_loss']*objectness_loss + loss_weights['box_loss']*box_loss + loss_weights['sem_cls_loss']*sem_cls_loss
    loss *= 10
    dict_output['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(dict_output['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    dict_output['obj_acc'] = obj_acc

    return loss, dict_output