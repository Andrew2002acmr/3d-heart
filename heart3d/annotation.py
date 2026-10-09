"""Original-grid research edits with explicit labels, undo and immutable versions."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from heart3d.storage import sha256_file, require_space

@dataclass
class MaskEditor:
    mask: np.ndarray
    affine: np.ndarray
    labels: dict
    source_ct_hash: str
    source_mask_hash: str
    physical_geometry_verified: bool = False
    def __post_init__(self):
        if not self.physical_geometry_verified:raise ValueError("Confirmed physical geometry required")
        self.mask=np.array(self.mask,copy=True);self.affine=np.asarray(self.affine,dtype=float)
        if self.mask.ndim!=3 or self.affine.shape!=(4,4) or not np.isfinite(self.affine).all():raise ValueError("Invalid original grid")
        if not np.allclose(self.affine[3],[0,0,0,1]) or abs(np.linalg.det(self.affine[:3,:3]))<1e-12:raise ValueError("Invalid affine")
        if not np.issubdtype(self.mask.dtype,np.integer) or self.mask.min()<0:raise ValueError("Integer labels required")
        if set(np.unique(self.mask))-({0}|set(self.labels)):raise ValueError("Unknown source label")
        self.undo_stack=[];self.operations=[]
    def world_to_voxel(self,point):
        world=np.asarray(point,dtype=float)
        if world.shape!=(3,) or not np.isfinite(world).all():raise ValueError("Invalid picked point")
        continuous=nib.affines.apply_affine(np.linalg.inv(self.affine),world)
        if np.any(continuous<-.5) or np.any(continuous>np.array(self.mask.shape)-.5):raise ValueError("Picked point outside CT")
        return np.clip(np.rint(continuous).astype(int),0,np.array(self.mask.shape)-1)
    def paint(self,axis,index,center,radius_mm,label):
        if axis not in (0,1,2) or not 0<=index<self.mask.shape[axis]:raise ValueError("Invalid slice")
        if label not in {0,*self.labels} or not np.isfinite(radius_mm) or radius_mm<=0:raise ValueError("Invalid brush")
        other=[i for i in range(3) if i!=axis];center=np.asarray(center,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all():raise ValueError("Invalid brush center")
        u,v=np.meshgrid(np.arange(self.mask.shape[other[0]]),np.arange(self.mask.shape[other[1]]),indexing="ij")
        delta=np.stack((u-center[0],v-center[1]),axis=-1)
        metric=self.affine[:3,other].T@self.affine[:3,other]
        chosen=np.einsum("...i,ij,...j->...",delta,metric,delta)<=radius_mm**2
        selection=[slice(None)]*3;selection[axis]=index;plane=self.mask[tuple(selection)]
        coords=np.flatnonzero(chosen & (plane!=label))
        if not len(coords):return 0
        old=plane.flat[coords].copy();self.undo_stack.append((axis,index,coords,old));plane.flat[coords]=label
        self.operations.append({"action":"brush","axis":axis,"slice":int(index),"center":center.tolist(),
                                "radius_mm":float(radius_mm),"label":int(label),"changed_voxels":len(coords)})
        return len(coords)
    def undo(self):
        if not self.undo_stack:return False
        axis,index,coords,old=self.undo_stack.pop();selection=[slice(None)]*3;selection[axis]=index
        self.mask[tuple(selection)].flat[coords]=old
        self.operations.append({"action":"undo","changed_voxels":len(coords)});return True
    def save(self,root,source_ct,source_mask,review_note="",reserve_GB=80):
        if sha256_file(source_ct)!=self.source_ct_hash or sha256_file(source_mask)!=self.source_mask_hash:raise ValueError("Source changed; version refused")
        require_space(root,self.mask.nbytes+20_000_000,int(reserve_GB*1e9))
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        version=root/datetime.now(timezone.utc).strftime("v_%Y%m%dT%H%M%S_%fZ");version.mkdir()
        image=nib.Nifti1Image(self.mask.astype(np.uint16),self.affine)
        image.header.set_xyzt_units("mm");image.set_qform(self.affine,1);image.set_sform(self.affine,1)
        path=version/"research_mask.nii.gz";nib.save(image,path)
        record={"annotation_status":"research_edited" if self.operations else "draft_prediction","expert_approval":False,
                "clinical_status":"unknown","source_CT_SHA256":self.source_ct_hash,"source_mask_SHA256":self.source_mask_hash,
                "labels":self.labels,"operations":self.operations,"review_note":review_note,
                "original_affine_ras":self.affine.tolist(),"original_shape":list(self.mask.shape),
                "resampled":False,"mask_SHA256":sha256_file(path)}
        (version/"annotation.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8");return version
