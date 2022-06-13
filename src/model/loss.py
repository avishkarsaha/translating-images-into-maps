import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import src
from torch.nn.utils.rnn import pad_sequence


def class_obj_size():
    """
    NuScenes class object size on Roddick Split
    gt_dir/name: semantic_maps_new_200x200
    """

    # Object size per class relative to map resolution
    s200 = [0.263, 0.031, 0.055, 0.029, 0.009, 0.002, 0.010, 0.001, 0.008, 0.006, 0.001]
    s100 = [0.296, 0.037, 0.066, 0.035, 0.011, 0.003, 0.013, 0.002, 0.010, 0.008, 0.002]
    s50 = [0.331, 0.046, 0.083, 0.044, 0.015, 0.004, 0.019, 0.003, 0.014, 0.011, 0.003]
    s25 = [0.380, 0.065, 0.114, 0.061, 0.022, 0.007, 0.031, 0.006, 0.022, 0.017, 0.008]
    s13 = [0.465, 0.105, 0.169, 0.093, 0.037, 0.016, 0.058, 0.013, 0.039, 0.031, 0.021]
    s7 = [0.599, 0.196, 0.279, 0.148, 0.069, 0.037, 0.117, 0.033, 0.076, 0.065, 0.054]

    ms_obj_size = torch.stack(
        [
            torch.tensor(s100),
            torch.tensor(s50),
            torch.tensor(s25),
            torch.tensor(s13),
            torch.tensor(s7),
        ]
    )

    return ms_obj_size.cuda()


def get_class_frequency():
    """
    NuScenes class frequency on Roddick Split
    gt_dir/name: semantic_maps_new_200x200
    """

    # Class frequency
    s200 = [
        0.999,
        0.489,
        0.969,
        0.294,
        0.094,
        0.080,
        0.758,
        0.0646,
        0.070,
        0.080,
        0.313,
        0.485,
    ]

    return torch.tensor(s200).cuda()


def class_weight():
    """
    NuScenes class frequency on Roddick Split
    gt_dir/name: semantic_maps_new_200x200
    """

    # Class weights
    w = torch.tensor([1.31, 5.43, 1.00, 2.13, 8.04, 1.19, 1.75, 5.71])

    return w


def class_flood_level():
    ms_flood = torch.tensor(
        [
            [
                0.0356,
                0.2656,
                0.1070,
                0.6454,
                0.8,
                0.75,
                0.1832,
                0.8,
                0.75,
                0.6633,
                0.5591,
            ],
            [
                0.0294,
                0.2418,
                0.0902,
                0.5982,
                0.75,
                0.7,
                0.1589,
                0.8,
                0.75,
                0.6302,
                0.5116,
            ],
            [
                0.0247,
                0.2363,
                0.0787,
                0.5765,
                0.75,
                0.75,
                0.1416,
                0.8,
                0.75,
                0.6317,
                0.4982,
            ],
        ]
    ).cuda()
    return ms_flood


class MultiTaskLoss(nn.Module):
    def __init__(self, l1_reduction="mean"):
        super().__init__()

        self.log_vars = nn.Parameter(torch.zeros(2))
        self.l1_reduction = l1_reduction

    def forward(self, preds, labels, bev, bev_pred):
        precision1 = torch.exp(-self.log_vars[0])
        precision2 = torch.exp(-self.log_vars[1])

        s1_loss = dice_loss_mean(preds[0], labels[0])
        s2_loss = dice_loss_mean(preds[1], labels[1])
        s4_loss = dice_loss_mean(preds[2], labels[2])

        dice_l = s1_loss + s2_loss + s4_loss
        bev_l = F.l1_loss(bev_pred, bev, reduction=self.l1_reduction)

        total_loss = (precision1 * dice_l + self.log_vars[0]) + (
            precision2 * bev_l + self.log_vars[1]
        )

        return (
            total_loss,
            float(self.log_vars[0]),
            float(self.log_vars[1]),
            float(dice_l),
            float(bev_l),
        )


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    logits = logits.view(-1)
    labels = labels.view(-1)
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def basic_weighted_dice_loss_mean(pred, label, idx_scale=None, args=None):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    # Set up class weighting
    weight = class_weight()

    loss_mean = (weight * (1 - iou)).mean()

    return loss_mean


def weighted_dice_loss_mean(pred, label, idx_scale, args):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    # Set up class weighting
    obj_size = class_obj_size()[idx_scale]
    class_freq = class_frequency()[idx_scale]
    weight = 1 / obj_size ** args.exp_os * 1 / class_freq ** args.exp_cf

    # Set large class' weights to one
    idxs_to_one = torch.tensor([0, 1, 2, 3, 6]).long()
    weight[idxs_to_one] = 1

    loss_mean = (weight * (1 - iou)).mean()

    return loss_mean


def weighted_dice_loss_sum(pred, label, idx_scale, args):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    # Set up class weighting
    obj_size = class_obj_size()[idx_scale]
    class_freq = class_frequency()[idx_scale]
    weight = 1 / obj_size ** args.exp_os * 1 / class_freq ** args.exp_cf

    loss_mean = (weight * (1 - iou)).sum()

    return loss_mean


def dice_loss_sum(pred, label, scale_idx=None, args=None):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    loss_mean = (1 - iou).sum()

    return loss_mean


def dice_loss_mean(pred, label, vis_mask=None, scale_idx=None, args=None):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    loss_mean = 1 - iou.mean()

    return loss_mean


def dice_loss_mean_wo_sigmoid(pred, label, vis_mask=None, scale_idx=None, args=None):
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum()) / (union.float().sum() + 1e-5)

    loss_mean = 1 - iou.mean()

    return loss_mean


def dice_loss_mean_vis_only(pred, label, vis_mask, scale_idx=None, args=None):
    # Calculates the loss in the visible areas only
    pred = torch.sigmoid(pred) * vis_mask.float()
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    loss_mean = 1 - iou.mean()

    return loss_mean


def dice2_loss_mean(pred, label, scale_idx=None, args=None):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1.0) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1.0
    )

    loss_mean = 1 - iou.mean()

    return loss_mean


def dice_loss_mean_mix1(pred, label, scale_idx=None, args=None):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label

    e_numerator = torch.zeros(pred.shape[1]) + 1e-5
    idxs_to_zero_out = torch.tensor([4, 5, 7, 8, 10]).long()
    e_numerator[idxs_to_zero_out] = 0
    e_numerator = e_numerator.cuda()

    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + e_numerator) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    loss_mean = 1 - iou.mean()

    return loss_mean


def dice_loss_mean_mix2(pred, label, scale_idx=None, args=None):
    pred = torch.sigmoid(pred)
    label = label.float()
    intersection = 2 * pred * label
    union = pred + label

    e_numerator = torch.zeros(pred.shape[1]) + 1e-5
    idxs_to_zero_out = torch.tensor([4, 5, 7, 8]).long()
    e_numerator[idxs_to_zero_out] = 0
    e_numerator = e_numerator.cuda()

    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + e_numerator) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )

    loss_mean = 1 - iou.mean()

    return loss_mean


def dice_loss_per_class(pred, labels, scale_idx=None):
    pred = torch.sigmoid(pred)
    labels = labels.float()
    intersection = 2 * pred * labels
    union = pred + labels
    iou = (intersection.float().sum(dim=0).sum(dim=-1).sum(dim=-1)) / (
        union.float().sum(dim=0).sum(dim=-1).sum(dim=-1) + 1e-5
    )
    return 1 - iou


def dice_loss_per_class_infer(pred, labels, scale_idx=None):
    pred = torch.sigmoid(pred)
    labels = labels.float()
    intersection = (2 * pred * labels).float().sum(dim=0).sum(dim=-1).sum(dim=-1)
    union = (pred + labels).float().sum(dim=0).sum(dim=-1).sum(dim=-1)
    iou = (intersection) / (union + 1e-5)
    return 1 - iou, intersection, union


def flooded_dice_loss_mean(dice_loss, scale_idx):

    # Flood level per class
    flood_level = class_flood_level()[scale_idx]

    flood_loss = (dice_loss - flood_level).abs() + flood_level

    return flood_loss.mean()


def flooded_dice_loss_sum(dice_loss, scale_idx):

    # Flood level per class
    flood_level = class_flood_level()[scale_idx]

    flood_loss = (dice_loss - flood_level).abs() + flood_level

    return flood_loss.sum()


def iou_loss(pred, labels):
    e = 1e-6
    pred = torch.sigmoid(pred)
    labels = labels.float()
    intersection = pred * labels
    union = (pred + labels) - (pred * labels)
    iou = intersection.sum() / (union.sum() + e)
    return 1.0 - iou


def bce_plus_dice_loss(pred, labels, weights, alpha):
    bce = F.binary_cross_entropy_with_logits(
        input=pred, target=labels, weight=weights, reduction="mean"
    )
    dice = dice_loss_mean(pred, labels)
    total = (alpha[0] * bce + alpha[1] * dice).sum()
    return total


def bce_plus_iou_loss(pred, labels, weights, alpha):
    bce = F.binary_cross_entropy_with_logits(
        input=pred, target=labels, weight=weights, reduction="mean"
    )
    iou = iou_loss(pred, labels)
    total = (alpha[0] * bce + alpha[1] * iou).sum()
    return total


def class_balanced_loss(
    labels, logits, samples_per_cls, no_of_classes, loss_type, alpha=None
):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    # effective_num = 1.0 - np.power(beta, samples_per_cls)
    # weights = (1.0 - beta) / np.array(effective_num)
    # weights = weights / np.sum(weights) * no_of_classes

    if loss_type == "bce":
        labels_one_hot = F.one_hot(labels.long(), no_of_classes).float()
        weights = 1 / np.array(samples_per_cls)
        weights = torch.tensor(weights).float().cuda()
        weights = weights.view(1, 1, 1, 1, -1)
        weights = (
            weights.repeat(
                labels_one_hot.shape[0],
                labels_one_hot.shape[1],
                labels_one_hot.shape[2],
                labels_one_hot.shape[3],
                1,
            )
            * labels_one_hot
        )
        weights = weights.sum(-1)
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels, weight=weights
        )
    elif loss_type == "iou":
        cb_loss = iou_loss(logits, labels)
    elif loss_type == "dice":
        cb_loss = dice_loss_mean(logits, labels)
    elif loss_type == "lovasz":
        cb_loss = lovasz_hinge_flat(logits, labels)
    # elif loss_type == "bce_iou":
    #     cb_loss = bce_plus_iou_loss(logits, labels, weights, alpha)
    # elif loss_type == "bce_dice":
    #     cb_loss = bce_plus_dice_loss(logits, labels, weights, alpha)

    return cb_loss


def weighted_bce_loss(pred, label, vis_mask):
    # Apply visibility mask to predictions
    pred = pred * vis_mask.float()

    pred = torch.sigmoid(pred)

    label = label.float()

    eps = 1e-12

    # Reshape weights for broadcasting
    cf = get_class_frequency()[None, :, None, None]
    pos_weights = torch.sqrt(1 / cf)
    neg_weights = torch.sqrt(1 / (1 - cf))

    # labels * -log(sigmoid(logits)) +
    # (1 - labels) * -log(1 - sigmoid(logits))

    loss = pos_weights * label * -torch.log(pred + eps) + (1 - label) * -torch.log(
        1 - pred + eps
    )

    return loss.sum(dim=-1).sum(dim=-1).sum(dim=0)


def uncertainty_loss(pred, vis_mask):
    # Invert visibility mask
    vis_mask = 1 - vis_mask.float()
    eps = 1e-12
    pred = pred * vis_mask
    pred = torch.sigmoid(pred)
    loss = 1 - pred * torch.log(pred + eps)

    return loss.sum(dim=-1).sum(dim=-1).sum(dim=0)


def aux_recognition_loss(pred, label):
    """
    BCE loss for class probabilities, multi-label classification
    """
    # pred = torch.sigmoid(pred)
    label = src.utils.count_classes_per_sample(label)

    assert pred.shape == label.shape

    class_freq = get_class_frequency()[None, :]
    weights = (1 / class_freq).repeat(len(pred), 1)

    loss = F.binary_cross_entropy_with_logits(
        pred, label, weight=weights, reduction="none"
    )
    return loss


def focal_loss(pred, gt, alpha, gamma):
    BCE_loss = F.binary_cross_entropy_with_logits(
        pred, gt.float(), reduction="none"
    ).sum(-1)
    gt, _ = torch.max(gt, dim=-1)
    gt = gt.long()
    at = torch.tensor(alpha, device=gt.device).gather(0, gt.data.view(-1))
    pt = torch.exp(-BCE_loss)
    F_loss = at * (1 - pt) ** gamma * BCE_loss
    return F_loss
