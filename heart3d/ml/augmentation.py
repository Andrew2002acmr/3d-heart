"""Small shared spatial transformations and intensity perturbations; no flips."""
import math
import torch
import torch.nn.functional as F


def augment(image,target,settings,generator):
    if settings.get('flips',False):raise ValueError('Flips require a separate orientation experiment')
    angle=(torch.rand((),generator=generator).item()*2-1)*settings['rotation_degrees']*math.pi/180
    scale=1+(torch.rand((),generator=generator).item()*2-1)*settings['scale_delta']
    if scale<=0:raise ValueError('Positive spatial scale required')
    c,s=math.cos(angle)/scale,math.sin(angle)/scale
    matrix=image.new_tensor([[c,-s,0],[s,c,0]]).unsqueeze(0)
    grid=F.affine_grid(matrix,(1,*image.shape),align_corners=False)
    # Input normalized air/pad=-1. Zero padding after +1 restores the same value.
    image=F.grid_sample((image+1).unsqueeze(0),grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]-1
    target=F.grid_sample(target.unsqueeze(0),grid,mode='nearest',padding_mode='zeros',align_corners=False)[0]
    shift=(torch.rand((),generator=generator).item()*2-1)*settings['intensity_shift']
    noise=torch.randn(image.shape,generator=generator,dtype=image.dtype)*settings['noise_std']
    return (image+shift+noise).clamp(-1,1),target
