"""Release-case 2.5D data in index coordinates, without invented HU or mm."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import torch
from torch.utils.data import Dataset
from heart3d.ml.augmentation import augment
from heart3d.ml.bundle import safe_relative
from heart3d.storage import sha256_file


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8", newline="\n")


def load_protocol(config):
    cohort=read_json(config["cohort"]); split=read_json(config["split"])
    if split["cohort_manifest_SHA256"] != sha256_file(config["cohort"]):
        raise ValueError("Frozen cohort changed")
    if not split.get("frozen"): raise ValueError("Frozen split required")
    rows={r["case_id"]:r for r in cohort["records"]}
    if len(rows)!=len(cohort["records"]): raise ValueError("Duplicate cohort case")
    seen=set(); groups={}; images={}
    for partition in ("train","validation","test"):
        if not split["partitions"][partition]: raise ValueError("Empty partition")
        for case in split["partitions"][partition]:
            if case in seen or case not in rows: raise ValueError("Case leakage or unknown case")
            row=rows[case]; group=row["group_id"]; image=row["image_SHA256"]
            for key, mapping in ((group, groups),(image, images)):
                if key in mapping and mapping[key]!=partition: raise ValueError("Group/image leakage")
                mapping[key]=partition
            seen.add(case)
    if seen!=set(rows): raise ValueError("Frozen split does not cover cohort")
    if set(split["development_cases"]) & set(split["partitions"]["test"]):
        raise ValueError("Development case in test")
    return rows, split


def index_transform(shape, size):
    shape=np.asarray(shape,dtype=int)
    if shape.shape!=(3,) or np.any(shape<2) or size<8: raise ValueError("Invalid index grid")
    resized=np.maximum(1,np.rint(size*shape[:2]/shape[:2].max()).astype(int))
    return {"original_shape_xyz":shape.tolist(), "size":int(size),
            "resized_xy":resized.tolist(), "pad_xy":((size-resized)//2).tolist(),
            "slice_count":int(shape[2]), "z_resampling":False,
            "coordinate_mapping":"pixel centers: (output+.5)*source/resized-.5",
            "array_axes":"network Z,Y,X; source X,Y,Z",
            "physical_geometry_used":False, "crop":"none; full-FOV index fit/pad"}


def resize_plane(plane, transform, order, fill):
    nx,ny,_=transform["original_shape_xyz"]; rx,ry=transform["resized_xy"]; px,py=transform["pad_xy"]
    if plane.shape!=(nx,ny) or order not in (0,1): raise ValueError("Plane/transform mismatch")
    gx,gy=np.meshgrid((np.arange(rx)+.5)*nx/rx-.5,(np.arange(ry)+.5)*ny/ry-.5,indexing="xy")
    output=np.full((transform["size"],transform["size"]),fill,dtype=np.float32)
    output[py:py+ry,px:px+rx]=map_coordinates(plane,[gx,gy],order=order,mode="nearest",prefilter=False)
    return output


def restore_probability_plane(probability, transform):
    """Restore C,H,W probabilities to C,X,Y; argmax is performed afterwards."""
    nx,ny,_=transform["original_shape_xyz"]; rx,ry=transform["resized_xy"]; px,py=transform["pad_xy"]
    size=transform["size"]
    if probability.ndim!=3 or probability.shape[1:]!=(size,size): raise ValueError("Prediction grid differs")
    x=(np.arange(nx)+.5)*rx/nx-.5; y=(np.arange(ny)+.5)*ry/ny-.5
    gx,gy=np.meshgrid(x,y,indexing="ij")
    return np.stack([map_coordinates(p[py:py+ry,px:px+rx],[gy,gx],order=1,mode="nearest",prefilter=False)
                     for p in probability])


def index_neighbors(center, count, offsets):
    if not 0<=center<count: raise ValueError("Center outside source")
    requested=center+np.asarray(offsets,dtype=int)
    return np.clip(requested,0,count-1), (requested<0)|(requested>=count)


class CardiacDataset(Dataset):
    def __init__(self, config_path, partition="train", data_root=None, augmentation=True):
        self.config=read_json(config_path); self.root=Path(data_root or self.config["data_root"])
        if partition not in self.config.get("allowed_partitions", ("train", "validation", "test")):
            raise ValueError("Partition is sealed for this experiment")
        rows,self.split=load_protocol(self.config)
        self.preproc=read_json(self.config["preprocessing"])
        if self.preproc["fitted_partition"]!="train" or self.preproc["split_SHA256"]!=sha256_file(self.config["split"]):
            raise ValueError("Preprocessing is not fitted to this train split")
        if self.preproc["cohort_SHA256"]!=sha256_file(self.config["cohort"]): raise ValueError("Cohort/preprocessing mismatch")
        self.partition=partition; self.rows=[rows[c] for c in self.split["partitions"][partition]]
        self.augmentation=bool(augmentation and partition=="train")
        self.provenance={}; self.opened=OrderedDict(); self.headers={}
        for row in self.rows:
            case=row["case_id"]; p=read_json(self.root/self.config["cache_relative_path"]/case/"provenance.json")
            if p["case_id"]!=case or p["partition"]!=partition: raise ValueError("Cache case/partition leakage")
            for key,path in (("split_SHA256",self.config["split"]),("preprocessing_SHA256",self.config["preprocessing"]),("cohort_SHA256",self.config["cohort"])):
                if p[key]!=sha256_file(path): raise ValueError("Stale cache provenance")
            if p["source_image_SHA256"]!=row["image_SHA256"] or p["source_mask_SHA256"]!=row["mask_SHA256"]:
                raise ValueError("Cache/source provenance differs")
            self.provenance[case]=p
        self.set_epoch(0)

    def set_epoch(self,epoch):
        self.epoch=epoch; self.indices=[]
        for row in self.rows:
            case=row["case_id"]; p=self.provenance[case]
            if self.partition=="train":
                seed=int.from_bytes(hashlib.sha256(f'{self.config["seed"]}|{epoch}|{case}'.encode()).digest()[:8],"little")
                negative=np.random.default_rng(seed).choice(p["negative_indices"],size=min(len(p["negative_indices"]),len(p["positive_indices"])),replace=False).tolist()
                centers=sorted(p["positive_indices"]+negative)
            else: centers=list(range(p["transform"]["slice_count"]))
            self.indices.extend((case,center) for center in centers)

    def arrays(self,case):
        if case not in self.provenance: raise ValueError("Case outside partition")
        if case not in self.opened:
            if case not in self.headers:
                directory=self.root/self.config["cache_relative_path"]/case; headers=[]
                try:
                    for name,expected in (("image.npy",np.dtype("float32")),("mask.npy",np.dtype("uint8"))):
                        stream=(directory/name).open("rb"); headers.append([stream])
                        version=np.lib.format.read_magic(stream)
                        reader={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}.get(version)
                        if reader is None: raise ValueError("Unsupported NPY header")
                        shape,fortran,dtype=reader(stream); t=self.provenance[case]["transform"]
                        if shape!=(t["slice_count"],t["size"],t["size"]) or dtype!=expected: raise ValueError("Cache header differs")
                        headers[-1].extend([dtype,shape,stream.tell(),"F" if fortran else "C"])
                except BaseException:
                    for h in headers: h[0].close()
                    raise
                self.headers[case]=headers
            self.opened[case]=tuple(np.memmap(handle,dtype=dtype,shape=shape,offset=offset,order=order,mode="r")
                                    for handle,dtype,shape,offset,order in self.headers[case])
            while len(self.opened)>2:
                _,old=self.opened.popitem(last=False)
                for array in old: array._mmap.close()
        self.opened.move_to_end(case)
        return self.opened[case]

    def __getitem__(self,index):
        case,center=self.indices[index]; image,mask=self.arrays(case)
        neighbors,outside=index_neighbors(center,len(image),self.preproc["context_offsets_indices"])
        x=torch.from_numpy(np.array(image[neighbors],dtype=np.float32))
        y=torch.from_numpy(np.array(mask[center],dtype=np.int64))
        if self.augmentation:
            generator=torch.Generator().manual_seed(self.config["seed"]+self.epoch*1000003+index)
            valid=y!=255; packed=torch.stack([y.masked_fill(~valid,0).float(),valid.float()])
            x,packed=augment(x,packed,self.config["augmentation"],generator)
            y=packed[0].round().long().masked_fill(packed[1]<.5,255)
        return {"image":x,"target":y,"case_id":case,"center_index":center,
                "context_outside_source":torch.from_numpy(outside)}

    def __len__(self): return len(self.indices)

    def close(self):
        for arrays in self.opened.values():
            for a in arrays:
                if not a._mmap.closed:a._mmap.close()
        for headers in self.headers.values():
            for h in headers:h[0].close()
        self.opened.clear(); self.headers.clear()

    def __del__(self):
        if hasattr(self,"headers"): self.close()


def confusion_counts(prediction,target,classes=8):
    p=np.asarray(prediction); t=np.asarray(target)
    if p.shape!=t.shape: raise ValueError("Prediction/target dimensions differ")
    valid=t!=255
    if np.any((p<0)|(p>=classes)) or np.any(valid & ((t<0)|(t>=classes))): raise ValueError("Unknown class")
    return np.bincount(classes*t[valid].astype(np.int64)+p[valid].astype(np.int64),minlength=classes**2).reshape(classes,classes)


def scores_from_confusion(counts):
    result={}; dice=[]
    names=("background","LV","RV","LA","RA","MYO","AO","PA")
    for c in range(1,8):
        tp=int(counts[c,c]); truth=int(counts[c].sum()); predicted=int(counts[:,c].sum())
        item={"target_voxels":truth,"predicted_voxels":predicted,"Dice":2*tp/(truth+predicted) if truth+predicted else None,
              "IoU":tp/(truth+predicted-tp) if truth+predicted-tp else None,
              "precision":tp/predicted if predicted else None,"recall":tp/truth if truth else None}
        result[names[c]]=item
        # Empty-reference classes are reported, never rewarded as perfect segmentation.
        if truth: dice.append(item["Dice"])
    return {"classes":result,"macro_foreground_Dice":float(np.mean(dice)) if dice else None,
            "macro_policy":"reference-present foreground classes only; no empty-empty reward"}
