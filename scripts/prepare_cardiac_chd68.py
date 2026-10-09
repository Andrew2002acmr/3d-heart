"""Freeze a public-release pilot and build its train-fitted index-grid cache on external storage."""
import argparse
import hashlib
from pathlib import Path
import shutil
import nibabel as nib
import numpy as np
from heart3d.ml.cardiac_data import read_json,write_json,load_protocol,index_transform,resize_plane
from heart3d.ml.bundle import pack,verify,safe_relative
from heart3d.storage import require_space,sha256_file

DEVELOPMENT={"ct_1001","ct_1002","ct_1003","ct_1004","ct_1008","ct_1072"}


def freeze(config_path,audit_path):
    config=read_json(config_path)
    if Path(config["cohort"]).exists() or Path(config["split"]).exists():raise ValueError("Protocol exists; never re-split v1")
    audit=read_json(audit_path); rows=[]
    for row in audit["cases"]:
        if row["pair_grid_gate"]!="passed":raise ValueError("Release pair gate failure requires review")
        case=row["case_id"]
        rows.append({"case_id":case,"group_id":row["duplicate_group"],"source":"CHD68",
                     "image_relative_path":f"public_chd/chd68/pairs/{case}_image.nii.gz",
                     "mask_relative_path":f"public_chd/chd68/pairs/{case}_label.nii.gz",
                     "image_SHA256":row["image_SHA256"],"mask_SHA256":row["mask_SHA256"],
                     "shape":row["shape"],"age_verified":False,"physical_scale_verified":False,
                     "diagnosis_codes":row["diagnosis_codes"],"healthy_status":row["healthy_status"]})
    if len(rows)!=68 or len({r["image_SHA256"] for r in rows})!=68:raise ValueError("Expected 68 distinct release image hashes")
    development=sorted(DEVELOPMENT & {r["case_id"] for r in rows})
    remaining=sorted(r["case_id"] for r in rows if r["case_id"] not in development)
    np.random.default_rng(config["seed"]).shuffle(remaining)
    train=sorted(development+remaining[:48-len(development)])
    rest=remaining[48-len(development):]
    cohort={"version":1,"scope":"public CHD68 release; not an age-verified pediatric cohort",
            "source_audit_SHA256":sha256_file(audit_path),"patient_identity_independently_verified":False,
            "case_group_policy":"one released case/group; CHD68 only; do not append ImageCHD duplicates",
            "local_NII_data_included":False,"records":rows}
    write_json(config["cohort"],cohort)
    split={"version":1,"seed":config["seed"],"cohort_manifest_SHA256":sha256_file(config["cohort"]),
           "split_unit":"released case/group; patient identity linkage not independently verified",
           "patient_level_independence_claimed":False,"development_cases":development,
           "partitions":{"train":train,"validation":sorted(rest[:10]),"test":sorted(rest[10:])},
           "frozen":True,"test_used_for_parameter_selection":False}
    write_json(config["split"],split);load_protocol(config)
    print("FROZEN_SPLIT",{p:len(ids) for p,ids in split["partitions"].items()},flush=True)


def prepare(config_path, data_root, statistics):
    config=read_json(config_path);root=Path(data_root); rows,split=load_protocol(config)
    cache=root/config["cache_relative_path"]
    if cache.exists() or Path(config["preprocessing"]).exists():raise ValueError("Prepared version exists; no implicit overwrite")
    required=sum(r["shape"][2]*config["input_size"]**2*5 for r in rows.values())
    require_space(root,required*2+2_000_000_000,int(80e9))
    cache.mkdir(parents=True);samples=[];records=[];prepared={}
    # TRAIN first. Neither validation nor test contributes to preprocessing fitting.
    for partition in ("train","validation","test"):
        if partition!="train" and not Path(config["preprocessing"]).exists():raise ValueError("Train fit missing")
        for case in split["partitions"][partition]:
            row=rows[case];ct_path=root/safe_relative(row["image_relative_path"]);mask_path=root/safe_relative(row["mask_relative_path"])
            if sha256_file(ct_path)!=row["image_SHA256"] or sha256_file(mask_path)!=row["mask_SHA256"]:raise ValueError("Source hash mismatch")
            ct=nib.load(ct_path);mask=nib.load(mask_path)
            if ct.shape!=mask.shape or not np.allclose(ct.affine,mask.affine,atol=1e-5,rtol=0):raise ValueError("Source pair grid mismatch")
            if tuple(ct.shape)!=tuple(row["shape"]) or nib.aff2axcodes(ct.affine)!=("R","A","S"):raise ValueError("Release orientation/shape requires review")
            image=ct.get_fdata(dtype=np.float32);target=np.asanyarray(mask.dataobj)
            if not np.isfinite(image).all() or not np.isfinite(target).all() or target.min()<0 or not np.equal(target,np.rint(target)).all():raise ValueError("Invalid source values")
            t=index_transform(ct.shape,config["input_size"]);directory=cache/case;directory.mkdir()
            shape=(t["slice_count"],t["size"],t["size"])
            x=np.lib.format.open_memmap(directory/"image.npy",mode="w+",dtype=np.float32,shape=shape)
            y=np.lib.format.open_memmap(directory/"mask.npy",mode="w+",dtype=np.uint8,shape=shape)
            positive=[];retained=np.zeros(8,dtype=np.int64)
            for z in range(shape[0]):
                x[z]=resize_plane(image[:,:,z],t,1,0)
                resized=resize_plane(target[:,:,z],t,0,255)
                y[z]=np.where(resized<=7,resized,255).astype(np.uint8)
                retained+=np.bincount(y[z][y[z]<8],minlength=8)
                if np.any((y[z]>=1)&(y[z]<=7)):positive.append(z)
            if np.any(retained[1:]==0):raise ValueError("A cardiac class vanished during preparation; review required")
            x.flush();y.flush();x._mmap.close();y._mmap.close()
            prepared[case]={"case_id":case,"partition":partition,"transform":t,"positive_indices":positive,
                            "negative_indices":sorted(set(range(shape[0]))-set(positive)),
                            "source_image_SHA256":row["image_SHA256"],"source_mask_SHA256":row["mask_SHA256"],
                            "original_affine":ct.affine.tolist(),"original_spatial_unit":ct.header.get_xyzt_units()[0],
                            "physical_geometry_verified":False}
            if partition=="train":
                seed=int.from_bytes(hashlib.sha256(case.encode()).digest()[:8],"little")^config["seed"]
                rng=np.random.default_rng(seed); coords=[rng.integers(0,n,size=65536) for n in image.shape]
                selected=image[tuple(coords)]; samples.append(selected)
                records.append({"case_id":case,"shape":list(image.shape),"intensity_min":float(image.min()),"intensity_max":float(image.max()),
                                "sample_percentiles":np.percentile(selected,[1,50,99.5]).tolist(),
                                "heart_fraction":float(((target>=1)&(target<=7)).sum()/target.size),
                                "positive_slices":len(positive),"negative_slices":shape[0]-len(positive),
                                "prepared_foreground_class_voxels":retained[1:].tolist()})
            del image,target
            print("PREPARED_RAW",partition,case,shape[0],flush=True)
        if partition=="train":
            low,high=np.percentile(np.concatenate(samples),[1,99.5])
            if high<=low:raise ValueError("Degenerate training distribution")
            pre={"version":1,"fitted_partition":"train","fit_cases":split["partitions"]["train"],
                 "split_SHA256":sha256_file(config["split"]),"cohort_SHA256":sha256_file(config["cohort"]),
                 "clip_source_intensity":[float(low),float(high)],"HU_claimed":False,
                 "normalization":"linear clip to [-1,1]; full-FOV padding=-1",
                 "fit_method":"65536 deterministic uniform original-grid samples per train case; equal case weight",
                 "percentile_probabilities":[1,99.5],"input_size":config["input_size"],"context_offsets_indices":[-2,-1,0,1,2],
                 "physical_context_used":False,"out_of_bounds_context":"replicate edge, explicitly flagged",
                 "image_interpolation":"linear XY, no Z resampling","mask_interpolation":"nearest XY, no Z resampling",
                 "ignore_index":255,"crop":"none; full-FOV index fit/pad; no GT-guided crop"}
            write_json(config["preprocessing"],pre)
            write_json(statistics,{"fitted_partition":"train","test_used":False,"cases":records,"preprocessing":pre})
            samples.clear()
    pre=read_json(config["preprocessing"]);low,high=pre["clip_source_intensity"]
    manifest=[]
    for partition in ("train","validation","test"):
        for case in split["partitions"][partition]:
            directory=cache/case; p=prepared[case];t=p["transform"]
            x=np.load(directory/"image.npy",mmap_mode="r+");rx,ry=t["resized_xy"];px,py=t["pad_xy"]
            for z in range(len(x)):
                plane=np.clip((x[z]-low)/(high-low),0,1)*2-1
                valid=np.zeros_like(plane,dtype=bool);valid[py:py+ry,px:px+rx]=True
                plane[~valid]=-1;x[z]=plane
            x.flush();x._mmap.close()
            p.update({"split_SHA256":sha256_file(config["split"]),"cohort_SHA256":sha256_file(config["cohort"]),
                      "preprocessing_SHA256":sha256_file(config["preprocessing"]),
                      "image_SHA256":sha256_file(directory/"image.npy"),"mask_SHA256":sha256_file(directory/"mask.npy")})
            write_json(directory/"provenance.json",p)
            for name in ("image.npy","mask.npy","provenance.json"):
                path=directory/name;manifest.append({"path":path.relative_to(root).as_posix(),"bytes":path.stat().st_size,"SHA256":sha256_file(path)})
    result={"version":1,"files":manifest,"source_bytes":sum(f["bytes"] for f in manifest),"raw_DICOM_included":False,
            "cohort_SHA256":sha256_file(config["cohort"]),"split_SHA256":sha256_file(config["split"]),
            "preprocessing_SHA256":sha256_file(config["preprocessing"]),"case_count":len(rows),"physical_geometry_verified":False}
    write_json(cache/"prepared_manifest.json",result)
    verify(root,cache/"prepared_manifest.json")
    print("CACHE_COMPLETE",result["case_count"],result["source_bytes"],flush=True)
    return result


def bundle(config_path,data_root,archive,manifest):
    config=read_json(config_path);root=Path(data_root);rows,split=load_protocol(config)
    receipt=verify(root,root/config["cache_relative_path"]/"prepared_manifest.json")
    paths=[r["path"] for r in receipt["files"]]+[config["cache_relative_path"]+"/prepared_manifest.json"]
    for partition in ("validation","test"):
        for case in split["partitions"][partition]:
            row=rows[case];path=row["mask_relative_path"]
            if sha256_file(root/path)!=row["mask_SHA256"]:raise ValueError("Original GT changed")
            paths.append(path)
    repository=[{"path":str(path).replace("\\","/"),"SHA256":sha256_file(path)} for path in
                (config_path,config["cohort"],config["split"],config["preprocessing"])]
    return pack(root,paths,archive,manifest,repository,int(80e9))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["freeze","prepare","bundle"])
    p.add_argument("--config",type=Path,required=True);p.add_argument("--data",type=Path)
    p.add_argument("--audit",type=Path);p.add_argument("--statistics",type=Path)
    p.add_argument("--archive",type=Path);p.add_argument("--manifest",type=Path)
    a=p.parse_args()
    if a.command=="freeze":freeze(a.config,a.audit)
    elif a.command=="prepare":prepare(a.config,a.data,a.statistics)
    else:bundle(a.config,a.data,a.archive,a.manifest)

if __name__=="__main__":main()
