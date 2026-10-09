"""Scratch multiclass CUDA benchmark/train/evaluate on a frozen public-release protocol."""
import argparse
import math
from pathlib import Path
import subprocess
import sys
import time
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from heart3d.ml.cardiac_data import (CardiacDataset,read_json,write_json,confusion_counts,
                                     scores_from_confusion,index_neighbors,restore_probability_plane)
from heart3d.ml.bundle import verify,safe_relative
from heart3d.ml.losses import MulticlassSegmentationLoss
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.resources import ResourceMonitor,environment
from heart3d.ml.runtime import revision,seed_runtime,step,experiment_paths
from heart3d.storage import require_space,sha256_file


def identity(config_path,config):
    return {"git_revision":revision(),"config_SHA256":sha256_file(config_path),
            "cohort_SHA256":sha256_file(config["cohort"]),"split_SHA256":sha256_file(config["split"]),
            "preprocessing_SHA256":sha256_file(config["preprocessing"]),"model_version":"own_cardiac_25d_unet_v1"}


def setup(config_path,data_root,device,output,require_cuda=True):
    config=read_json(config_path);root=Path(data_root or config["data_root"]); output=Path(output)
    if output.exists():raise ValueError("Run output exists; no implicit overwrite/resume")
    if require_cuda:
        if device!="cuda" or not torch.cuda.is_available():raise ValueError("Verified CUDA required; no CPU fallback")
        if "4090" not in torch.cuda.get_device_name(0):raise ValueError("Authorized RTX 4090 required")
        if not root.resolve().is_relative_to(Path("/workspace")) or not output.resolve().is_relative_to(Path("/workspace")):
            raise ValueError("RunPod data and outputs must stay under /workspace")
        if not Path.cwd().resolve().is_relative_to(Path("/workspace")):raise ValueError("RunPod repository must stay under /workspace")
        if subprocess.check_output(["git","status","--porcelain"],text=True).strip():raise ValueError("Commit code before GPU benchmark")
    if config["model"].get("output_channels")!=8 or config["loss"].get("ignore_index")!=255:
        raise ValueError("Explicit eight-class namespace and ignore_index=255 required")
    if config["num_workers"]!=0:raise ValueError("Explicit epoch state requires workers=0 in v1")
    seed_runtime(config["seed"],config["cpu_threads"],device)
    require_space(root,1_000_000_000,int(config["reserve_GB"]*1e9))
    manifest=root/config["cache_relative_path"]/"prepared_manifest.json"
    verified=verify(root,manifest)
    for key,path in (("cohort_SHA256",config["cohort"]),("split_SHA256",config["split"]),("preprocessing_SHA256",config["preprocessing"])):
        if verified[key]!=sha256_file(path):raise ValueError("Prepared cache belongs to another protocol")
    return config,root,{**identity(config_path,config),"prepared_manifest_SHA256":sha256_file(manifest)}


def sync(device):
    if device=="cuda":torch.cuda.synchronize()


def device_info(device):
    info=environment(device)
    if device=="cuda":
        info.update(GPU_name=torch.cuda.get_device_name(0),CUDA_version=torch.version.cuda,
                    GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    TF32=False,precision="FP32",deterministic_cuda="warn_only")
    return info


def gpu_memory(device):
    return {"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()} if device=="cuda" else None


@torch.no_grad()
def fixed_loss(model,criterion,batch,device):
    model.eval()
    return float(criterion(model(batch["image"].to(device)),batch["target"].to(device)))


def benchmark(config_path,output,data_root=None,batches=50,device="cuda",require_cuda=True):
    if not 1<=batches<=200:raise ValueError("Benchmark limited to 1..200 batches")
    config,root,info=setup(config_path,data_root,device,output,require_cuda);output=Path(output)
    dataset=CardiacDataset(config_path,"train",root);sanity=CardiacDataset(config_path,"train",root,augmentation=False)
    loader=DataLoader(dataset,batch_size=config["batch_size"],shuffle=True,num_workers=0,
                      pin_memory=device=="cuda",generator=torch.Generator().manual_seed(config["seed"]))
    warmup=config["benchmark_warmup_batches"]
    if not 0<=warmup<=20 or not 1<=config["sanity_batches"]<=50 or batches+warmup>len(loader):
        raise ValueError("Bounded benchmark exceeds configured limits or one epoch")
    model=HeartUNet25D(**config["model"]).to(device);criterion=MulticlassSegmentationLoss(**config["loss"])
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    cases=list(sanity.provenance)[:2];indices=[]
    for case in cases:
        positive=sanity.provenance[case]["positive_indices"]
        indices.append(sanity.indices.index((case,positive[len(positive)//2])))
    fixed=default_collate([sanity[i] for i in indices]);before=fixed_loss(model,criterion,fixed,device)
    initial=[p.detach().clone() for p in model.parameters()]
    if device=="cuda":torch.cuda.reset_peak_memory_stats()
    history=[];timings=[];samples=0;monitor=ResourceMonitor();iterator=iter(loader)
    output.mkdir(parents=True)
    with monitor:
        for i in range(batches+warmup):
            sync(device);start=time.perf_counter();batch=next(iterator)
            loss,norm=step(model,optimizer,criterion,batch,device);sync(device)
            elapsed=time.perf_counter()-start
            history.append({"step":i,"warmup":i<warmup,"seconds_including_data":elapsed,"loss":loss,"gradient_norm":norm,"case_ids":batch["case_id"]})
            if i>=warmup:timings.append(elapsed);samples+=len(batch["image"])
            if (i+1)%10==0:print("CARDIAC_BENCHMARK",i+1,round(elapsed,4),round(loss,5),flush=True)
        for _ in range(config["sanity_batches"]):step(model,optimizer,criterion,fixed,device)
        after=fixed_loss(model,criterion,fixed,device)
    delta=sum(float((p.detach()-old).abs().sum()) for p,old in zip(model.parameters(),initial))
    seconds=float(np.mean(timings));epoch_batches=math.ceil(len(dataset)/config["batch_size"])
    result={**info,"scope":"train-only bounded benchmark; not quality results","device":device,"GPU_name":torch.cuda.get_device_name(0) if device=="cuda" else None,
            "parameters":sum(p.numel() for p in model.parameters()),"warmup_batches":warmup,"measured_batches":batches,
            "mean_seconds_per_batch":seconds,"median_seconds_per_batch":float(np.median(timings)),"samples_per_second":samples/sum(timings),
            "train_samples_per_epoch":len(dataset),"train_batches_per_epoch":epoch_batches,
            "estimated_train_epoch_seconds":epoch_batches*seconds,"estimated_30_train_epochs_seconds":30*epoch_batches*seconds,
            "estimate_scope":"measured train batches extrapolated; excludes validation/checkpoints and contention",
            "fixed_train_case_ids":cases,"fixed_loss_before":before,"fixed_loss_after_sanity":after,
            "weights_absolute_delta":delta,"finite_losses_and_gradients":True,"sanity_passed":bool(after<before and delta>0),
            "resources":monitor.summary(),"GPU_memory":gpu_memory(device),"validation_used":False,"test_used":False,"full_training_started":False}
    write_json(output/"summary.json",result);write_json(output/"history.json",history)
    write_json(output/"environment.json",device_info(device));write_json(output/"config.json",config)
    write_json(output/"resource_samples.json",monitor.samples)
    dataset.close();sanity.close()
    print("BENCHMARK_RESULT",{k:result[k] for k in ("device","mean_seconds_per_batch","estimated_train_epoch_seconds","sanity_passed")},flush=True)
    if not result["sanity_passed"]:raise ValueError("Train-only learning sanity failed; investigate before full training")
    return result


def check_benchmark(receipt,info,device):
    report=read_json(receipt)
    if device!="cuda" or not torch.cuda.is_available():raise ValueError("Verified CUDA benchmark required")
    if not report["sanity_passed"] or report["device"]!="cuda" or report["GPU_name"]!=torch.cuda.get_device_name(0):
        raise ValueError("Successful benchmark on this GPU required")
    for key in info:
        if report[key]!=info[key]:raise ValueError("Code/config/data changed after benchmark")


@torch.no_grad()
def probability_batches(model,image,offsets,batch_size,device):
    model.eval()
    for start in range(0,len(image),batch_size):
        centers=list(range(start,min(start+batch_size,len(image))))
        contexts=np.stack([image[index_neighbors(center,len(image),offsets)[0]] for center in centers]).astype(np.float32)
        logits=model(torch.from_numpy(contexts).to(device))
        yield centers,logits.softmax(1).cpu().numpy()


def validation_score(model,dataset,config,device):
    records=[]
    for row in dataset.rows:
        case=row["case_id"];image,target=dataset.arrays(case);counts=np.zeros((8,8),dtype=np.int64)
        for centers,probability in probability_batches(model,image,dataset.preproc["context_offsets_indices"],config["batch_size"],device):
            counts+=confusion_counts(probability.argmax(1),target[centers])
        records.append({"case_id":case,**scores_from_confusion(counts)})
    return float(np.mean([r["macro_foreground_Dice"] for r in records])),records


def train(config_path,run_name,benchmark_path,data_root=None,allow_full_training=False,device="cuda",require_cuda=True):
    config=read_json(config_path);root,output,checkpoints=experiment_paths(config,run_name,data_root)
    if not allow_full_training or not 1<=config["epochs"]<=30:raise ValueError("Explicit full-training flag and 1..30 epochs required")
    if checkpoints.exists():raise ValueError("Checkpoint output exists")
    config,root,info=setup(config_path,root,device,output,require_cuda)
    if require_cuda:check_benchmark(benchmark_path,info,device)
    training=CardiacDataset(config_path,"train",root);validation=CardiacDataset(config_path,"validation",root,augmentation=False)
    model=HeartUNet25D(**config["model"]).to(device);criterion=MulticlassSegmentationLoss(**config["loss"])
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    output.mkdir(parents=True);checkpoints.mkdir(parents=True)
    write_json(output/"config.json",config);write_json(output/"environment.json",device_info(device))
    write_json(output/"run_provenance.json",{**info,"initialization":"scratch; benchmark weights never loaded",
               "benchmark_SHA256":sha256_file(benchmark_path) if benchmark_path else None,"test_used":False,"explicit_full_training_flag":True,
               "selection":"maximum mean release-case foreground validation Dice; preprocessing grid",
               "age_verified_pediatric_claim":False,"patient_identity_independently_verified":False})
    with (output/"pip_freeze.txt").open("w",encoding="utf-8") as stream:
        subprocess.run([sys.executable,"-m","pip","freeze"],stdout=stream,check=True)
    if device=="cuda":torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();history=[];best=-1.;monitor=ResourceMonitor()
    with monitor:
        for epoch in range(config["epochs"]):
            training.set_epoch(epoch);tick=time.perf_counter();total=0.;count=0
            loader=DataLoader(training,batch_size=config["batch_size"],shuffle=True,num_workers=0,pin_memory=device=="cuda",
                              generator=torch.Generator().manual_seed(config["seed"]+epoch))
            for batch in loader:
                loss,_=step(model,optimizer,criterion,batch,device);total+=loss*len(batch["image"]);count+=len(batch["image"])
            sync(device);train_seconds=time.perf_counter()-tick;tick=time.perf_counter()
            score,records=validation_score(model,validation,config,device);sync(device)
            record={"epoch":epoch+1,"training_loss":total/count,"validation_mean_case_foreground_Dice":score,
                    "training_seconds":train_seconds,"validation_seconds":time.perf_counter()-tick,"samples":count,"validation_cases":records}
            history.append(record);is_best=score>best;best=max(best,score)
            checkpoint={**info,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":config,
                        "preprocessing":training.preproc,"epoch":epoch+1,"best_validation_Dice":best,
                        "training_scope":"public_chd68_multiclass_scratch_v1","torch_rng_state":torch.get_rng_state(),"history":history}
            if is_best:torch.save(checkpoint,checkpoints/"best.pt")
            torch.save(checkpoint,checkpoints/"last.pt");write_json(output/"history.json",history)
            print("EPOCH",epoch+1,"loss",round(record["training_loss"],5),"val_macro_Dice",round(score,5),"train_sec",round(train_seconds,1),flush=True)
    result={**info,"epochs":len(history),"seconds":time.perf_counter()-start,"best_validation_Dice_prepared_grid":best,
            "best_epoch":int(np.argmax([r["validation_mean_case_foreground_Dice"] for r in history]))+1,
            "resources":monitor.summary(),"GPU_memory":gpu_memory(device),"test_evaluated":False,
            "checkpoints":{name:{"bytes":(checkpoints/name).stat().st_size,"SHA256":sha256_file(checkpoints/name)} for name in ("best.pt","last.pt")}}
    write_json(output/"training_summary.json",result);write_json(output/"resource_samples.json",monitor.samples)
    training.close();validation.close();return result


def load_checkpoint(path,config,device):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["training_scope"]!="public_chd68_multiclass_scratch_v1":raise ValueError("Not a trained multiclass checkpoint")
    for key,source in (("config_SHA256",config["_config_path"]),("split_SHA256",config["split"]),("preprocessing_SHA256",config["preprocessing"]),("cohort_SHA256",config["cohort"])):
        if checkpoint[key]!=sha256_file(source):raise ValueError("Checkpoint protocol differs")
    model=HeartUNet25D(**config["model"]).to(device);model.load_state_dict(checkpoint["model"]);return model,checkpoint


def evaluate(config_path,checkpoint_path,output,partition="test",data_root=None,device="cuda",require_cuda=True):
    config,root,info=setup(config_path,data_root,device,output,require_cuda);config["_config_path"]=str(config_path)
    dataset=CardiacDataset(config_path,partition,root,augmentation=False)
    model,checkpoint=load_checkpoint(checkpoint_path,config,device);output=Path(output);output.mkdir(parents=True)
    records=[]
    for row in dataset.rows:
        case=row["case_id"];image,_=dataset.arrays(case);p=dataset.provenance[case]
        gt_path=root/safe_relative(row["mask_relative_path"])
        if sha256_file(gt_path)!=row["mask_SHA256"]:raise ValueError("Original GT changed")
        gt=nib.load(gt_path);original=np.asanyarray(gt.dataobj);prediction=np.zeros(gt.shape,dtype=np.uint8);counts=np.zeros((8,8),dtype=np.int64)
        if gt.shape!=tuple(p["transform"]["original_shape_xyz"]) or not np.allclose(gt.affine,p["original_affine"],atol=1e-5,rtol=0):raise ValueError("Original GT grid differs")
        for centers,probability in probability_batches(model,image,dataset.preproc["context_offsets_indices"],config["batch_size"],device):
            for center,plane in zip(centers,probability):
                prediction[:,:,center]=restore_probability_plane(plane,p["transform"]).argmax(0).astype(np.uint8)
                target=np.where(original[:,:,center]<=7,original[:,:,center].astype(np.uint8),255).astype(np.uint8)
                counts+=confusion_counts(prediction[:,:,center],target)
        directory=output/case;directory.mkdir();saved=nib.Nifti1Image(prediction,gt.affine,gt.header.copy())
        saved.set_data_dtype(np.uint8);saved.header.set_slope_inter(1,0)
        nib.save(saved,directory/"cardiac_prediction_original.nii.gz")
        write_json(directory/"provenance.json",{**info,"checkpoint_SHA256":sha256_file(checkpoint_path),"case_id":case,
                   "source_mask_SHA256":row["mask_SHA256"],"original_affine":gt.affine.tolist(),"original_shape":list(gt.shape),
                   "physical_geometry_verified":False,"spatial_units_claimed_mm":False,
                   "prediction_SHA256":sha256_file(directory/"cardiac_prediction_original.nii.gz"),"GT_overwritten":False})
        records.append({"case_id":case,**scores_from_confusion(counts)})
        if sha256_file(gt_path)!=row["mask_SHA256"]:raise ValueError("GT integrity failed")
        print("EVALUATED",partition,case,flush=True)
    result={**info,"partition":partition,"checkpoint_SHA256":sha256_file(checkpoint_path),"checkpoint_epoch":checkpoint["epoch"],
            "grid":"original release NIfTI grid, probabilities restored before argmax","case_count":len(records),"records":records,
            "mean_case_macro_foreground_Dice":float(np.mean([r["macro_foreground_Dice"] for r in records])),
            "per_class_mean_Dice":{name:float(np.mean([r["classes"][name]["Dice"] for r in records if r["classes"][name]["target_voxels"]]))
                                   for name in ("LV","RV","LA","RA","MYO","AO","PA")},
            "HD95_ASSD_mm_calculated":False,"age_verified_pediatric_claim":False,
            "patient_identity_independently_verified":False,"external_NII_evaluation":False}
    write_json(output/"metrics.json",result);dataset.close();return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=["benchmark","train","evaluate"])
    p.add_argument("--config",type=Path,required=True);p.add_argument("--data",type=Path)
    p.add_argument("--output",type=Path);p.add_argument("--batches",type=int,default=50)
    p.add_argument("--run-name");p.add_argument("--benchmark",type=Path);p.add_argument("--allow-full-training",action="store_true")
    p.add_argument("--checkpoint",type=Path);p.add_argument("--partition",choices=["validation","test"],default="test")
    a=p.parse_args()
    needed={"benchmark":["output"],"train":["run_name","benchmark"],"evaluate":["checkpoint","output"]}[a.command]
    for name in needed:
        if getattr(a,name) is None:p.error("--"+name.replace("_","-")+" is required for "+a.command)
    if a.command=="benchmark":benchmark(a.config,a.output,a.data,a.batches)
    elif a.command=="train":train(a.config,a.run_name,a.benchmark,a.data,a.allow_full_training)
    else:evaluate(a.config,a.checkpoint,a.output,a.partition,a.data)

if __name__=="__main__":main()
