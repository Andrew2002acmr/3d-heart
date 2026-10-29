"""Reuse HeartUNet25D and add only a small coarse localizer/optional core branch."""
import torch
from torch import nn
from torch.nn import functional as F
from heart3d.ml.model import HeartUNet25D


def block3(a,b):return nn.Sequential(nn.Conv3d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU(),nn.Conv3d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU())


class HeartLocalizer(nn.Module):
    def __init__(self):
        super().__init__();self.a=block3(1,4);self.b=block3(4,8);self.c=block3(8,16);self.d=block3(24,8);self.e=block3(12,4);self.out=nn.Conv3d(4,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.max_pool3d(a,2));c=self.c(F.max_pool3d(b,2));d=self.d(torch.cat([F.interpolate(c,size=b.shape[2:],mode='trilinear',align_corners=False),b],1))
        return self.out(self.e(torch.cat([F.interpolate(d,size=a.shape[2:],mode='trilinear',align_corners=False),a],1)))


class ROISegmenter(nn.Module):
    def __init__(self,branch=False):
        super().__init__();self.base=HeartUNet25D(input_channels=5,encoder_channels=(16,32,64,128),norm_groups=8,output_channels=8);self.branch=branch
        if branch:self.core=nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.GroupNorm(8,16),nn.ReLU(),nn.Conv2d(16,16,1))
    def forward(self,x,core=None):
        if not self.branch:return self.base(x)
        if core is None or core.shape[1]!=4:raise ValueError('Four CT-only 3D-core channels required')
        image=self.base.encoder[0](x)+self.core(core);skips=[image]
        for i,layer in enumerate(self.base.encoder[1:],1):
            image=layer(self.base.pool(image))
            if i<3:skips.append(image)
        for layer,skip in zip(self.base.decoder,reversed(skips)):
            image=layer(torch.cat([F.interpolate(image,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip],1))
        return self.base.output(image)


def localizer_loss(logits,target):
    valid=target!=255;safe=target.masked_fill(~valid,0).float();p=logits[:,0].float().sigmoid()*valid
    ce=F.binary_cross_entropy_with_logits(logits[:,0].float()[valid],safe[valid])
    dice=1-(2*(p*safe).sum((1,2,3))+1e-6)/(p.sum((1,2,3))+(safe*valid).sum((1,2,3))+1e-6)
    return ce+dice.mean()
