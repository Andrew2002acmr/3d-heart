"""Bounded real-cache forward/backward smoke check; CPU, not a resource estimate."""
import argparse
from pathlib import Path
import torch
from torch.utils.data._utils.collate import default_collate
from heart3d.ml.cardiac_data import CardiacDataset,write_json
from heart3d.ml.losses import MulticlassSegmentationLoss
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.runtime import seed_runtime,step
from heart3d.storage import sha256_file


def run(config,data,output):
    output=Path(output)
    if output.exists():raise ValueError("Smoke report exists")
    seed_runtime(20261009,4,"cpu")
    dataset=CardiacDataset(config,"train",data,augmentation=False)
    cases=list(dataset.provenance)[:2];indices=[]
    for case in cases:
        positive=dataset.provenance[case]["positive_indices"]
        indices.append(dataset.indices.index((case,positive[len(positive)//2])))
    batch=default_collate([dataset[i] for i in indices])
    model=HeartUNet25D(**dataset.config["model"]);criterion=MulticlassSegmentationLoss(**dataset.config["loss"])
    optimizer=torch.optim.Adam(model.parameters(),lr=dataset.config["learning_rate"])
    initial=[p.detach().clone() for p in model.parameters()]
    history=[]
    for _ in range(3):
        loss,gradient=step(model,optimizer,criterion,batch,"cpu")
        history.append({"loss":loss,"gradient_norm":gradient})
    changed=any(not torch.equal(p.detach(),old) for p,old in zip(model.parameters(),initial))
    if not changed:raise ValueError("Weights unchanged")
    report={"scope":"real train cache technical smoke only; no GPU timing or accuracy",
            "train_case_ids":cases,"batches":3,"batch_size":2,"input_shape":list(batch["image"].shape),
            "output_shape":list(model(batch["image"]).shape),"target_shape":list(batch["target"].shape),
            "history":history,"weights_updated":changed,"finite_losses_gradients":True,
            "validation_test_used":False,"checkpoint_saved":False,
            "config_SHA256":sha256_file(config),"GPU_benchmark_executed":False}
    write_json(output,report);dataset.close();print("REAL_CACHE_SMOKE_PASSED",cases,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("config","data","output"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();run(a.config,a.data,a.output)
