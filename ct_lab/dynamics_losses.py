"""Simple preregistered loss ablations. Weights come exclusively from train targets."""
from torch import nn
from torch.nn import functional as F
from ct_lab.dynamics_core import *
from heart3d.ml.losses import MulticlassSegmentationLoss,multiclass_dice_loss

def train_weights(cases,statistics):
    split,_=cohort()
    if not set(cases)<=set(split['train']):raise ValueError('Loss weighting must be train only')
    lookup={r['case']:r for r in statistics};counts=np.array([lookup[c]['prepared_class_counts'] for c in cases],np.int64).sum(0)
    frequency=counts/max(1,counts.sum());weights=1/np.sqrt(np.maximum(frequency,1e-12));weights/=weights.mean();weights=np.clip(weights,.25,4).astype(np.float32)
    return {'train_cases':list(cases),'counts':counts.tolist(),'weights':weights.tolist(),'formula':'inverse sqrt unique prepared train voxel frequency; unit mean then clip[.25,4]; ignore255 excluded'}

class AdditionalLoss(nn.Module):
    def __init__(self,variant,weights=None,gamma=2.):
        super().__init__();self.variant=variant;self.gamma=gamma;self.register_buffer('weights',torch.tensor(weights,dtype=torch.float32) if weights is not None else None)
        if variant not in ('weighted_ce_dice','focal_dice'):raise ValueError('Unknown added loss')
    def forward(self,logits,target):
        valid=target!=255
        if not valid.any():return logits.sum()*0
        if self.variant=='weighted_ce_dice':ce=F.cross_entropy(logits,target,weight=self.weights,ignore_index=255)
        else:
            raw=F.cross_entropy(logits,target,ignore_index=255,reduction='none');ce=(((1-torch.exp(-raw[valid]))**self.gamma)*raw[valid]).mean()
        return ce+multiclass_dice_loss(logits,target,255)

def criterion(variant,cases,device):
    if variant in ('ce','ce_dice'):return MulticlassSegmentationLoss(variant=variant).to(device),None
    if variant=='weighted_ce_dice':
        fitted=train_weights(cases,json.loads((OUT/'case_statistics.json').read_text()));return AdditionalLoss(variant,fitted['weights']).to(device),fitted
    if variant=='focal_dice':return AdditionalLoss(variant,gamma=2.).to(device),None
    raise ValueError('Unknown preregistered loss')
