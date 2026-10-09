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

def multiclass_dice_loss(logits, target, ignore_index=255, include_background=False, epsilon=1e-6):
    """Mean class Dice across B,H,W; unspecified/noncardiac labels are ignored."""
    if logits.ndim!=4 or logits.shape[1]<2 or target.shape!=(logits.shape[0],*logits.shape[2:]) or target.dtype!=torch.long:
        raise ValueError('Multiclass logits BCHW and integer target BHW required')
    valid=target!=ignore_index
    if torch.any(valid & ((target<0)|(target>=logits.shape[1]))):
        raise ValueError('Target outside explicit class namespace')
    if not valid.any():return logits.sum()*0
    safe=target.masked_fill(~valid,0)
    truth=F.one_hot(safe,logits.shape[1]).permute(0,3,1,2).to(logits.dtype)*valid[:,None]
    probability=logits.softmax(dim=1)*valid[:,None]
    dice=(2*(probability*truth).sum((0,2,3))+epsilon)/(probability.sum((0,2,3))+truth.sum((0,2,3))+epsilon)
    return 1-dice[int(not include_background):].mean()

class MulticlassSegmentationLoss(nn.Module):
    def __init__(self,variant='ce_dice',ignore_index=255):
        super().__init__()
        if variant not in {'ce','dice','ce_dice'}:raise ValueError('Unknown multiclass loss')
        self.variant,self.ignore_index=variant,ignore_index
    def forward(self,logits,target):
        dice=multiclass_dice_loss(logits,target,self.ignore_index)
        if self.variant=='dice':return dice
        ce=F.cross_entropy(logits,target,ignore_index=self.ignore_index) if (target!=self.ignore_index).any() else logits.sum()*0
        return ce if self.variant=='ce' else ce+dice
