"""Binary segmentation losses implemented inside this project."""
import torch
from torch import nn
import torch.nn.functional as F


def soft_dice_loss(logits, target, epsilon=1e-6):
    if logits.shape!=target.shape or logits.ndim!=4 or logits.shape[1]!=1:
        raise ValueError('Matching [batch,1,height,width] binary tensors required')
    probability=logits.sigmoid()
    axes=(1,2,3)
    numerator=2*(probability*target).sum(axes)+epsilon
    denominator=probability.sum(axes)+target.sum(axes)+epsilon
    return (1-numerator/denominator).mean()


class SegmentationLoss(nn.Module):
    def __init__(self,variant='bce_dice',bce_weight=1.,dice_weight=1.):
        super().__init__()
        if variant not in {'bce','dice','bce_dice'}:raise ValueError('Unknown loss variant')
        self.variant=variant;self.bce_weight=bce_weight;self.dice_weight=dice_weight

    def forward(self,logits,target):
        if logits.shape!=target.shape:raise ValueError('Target/logit shape mismatch')
        bce=F.binary_cross_entropy_with_logits(logits,target)
        if self.variant=='bce':return bce
        dice=soft_dice_loss(logits,target)
        if self.variant=='dice':return dice
        return self.bce_weight*bce+self.dice_weight*dice
