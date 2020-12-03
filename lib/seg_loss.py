import torch
import torch.nn as nn
from torch.nn.modules import BCELoss


class SegLoss(nn.Module):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.bce_loss = BCELoss()

    def forward(self, pred_mask, real_mask):
        bce_loss = self.bce_loss(pred_mask, real_mask)
        intersection = torch.sum(pred_mask * real_mask) + 1e-7
        union = torch.sum(pred_mask + real_mask) - intersection + 1e-7
        iou = torch.div(intersection, union)
        loss = bce_loss - iou + 1
        return bce_loss, iou, loss
