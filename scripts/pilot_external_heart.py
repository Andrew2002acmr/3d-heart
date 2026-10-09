"""Verified local CT -> supplied project Heart OAR checkpoint -> original-grid draft."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import shutil
import time
import nibabel as nib
import numpy as np
from heart3d.ml.infer_dicom import infer_dicom
from heart3d.storage import require_space, sha256_file
from heart3d.surfaces import build_surface

def pilot(a):
    output=Path(a.output)
    if output.exists():raise ValueError("Pilot output exists; choose a new version")
    require_space(output,2_000_000_000,int(a.reserve_GB*1e9))
    audit=Path(a.audit_root)
    locators=json.loads((audit/"source_locators.private.json").read_text())
    row=locators["series"][a.series];source_root=Path(locators["input_root"])
    conversion=json.loads((audit/"prepared"/a.series/"conversion.json").read_text())
    if not conversion["geometry"]["physical_geometry_confirmed_from_DICOM"]:raise ValueError("DICOM physical gate required")
    output.mkdir(parents=True);ct_directory=output/"ct_series";ct_directory.mkdir();checks=[]
    for index,(relative,expected) in enumerate(zip(row["relative_paths"],row["source_hashes"],strict=True)):
        source=source_root/relative
        if sha256_file(source)!=expected:raise ValueError("Original DICOM hash changed")
        target=ct_directory/f"{index:04d}.dcm";shutil.copy2(source,target)
        if sha256_file(target)!=expected:raise ValueError("Staged DICOM hash mismatch")
        checks.append({"source_relative_path":relative,"SHA256":expected})
    start=time.perf_counter()
    with (output/"inference.private.log").open("w",encoding="utf-8") as log,redirect_stdout(log):
        infer_dicom(a.config,a.checkpoint,ct_directory,output/"prediction",a.device,
                    row["series_uid"],conversion["geometry"]["sop_set_sha256"],a.reserve_GB)
    image=nib.load(output/"prediction/heart_prediction_original.nii.gz");mask=np.asanyarray(image.dataobj)
    mesh,metrics=build_surface(mask,1,image.affine)
    if mesh is not None:
        metrics.update({"short_name":"Heart","name":"Heart OAR draft","annotation_status":"draft_prediction",
                        "voxel_volume_mm3":metrics["voxel_volume_in_coordinate_units_cubed"],
                        "surface_area_mm2":metrics["surface_area_in_coordinate_units_squared"]})
        mesh.field_data["coordinate_system"]=["RAS"];mesh.field_data["unit"]=["mm"];mesh.save(output/"heart_draft.vtp")
    unchanged=all(sha256_file(source_root/r["source_relative_path"])==r["SHA256"] for r in checks)
    if not unchanged:raise ValueError("Original integrity failed after inference")
    summary={"series_alias":a.series,"scope":"binary Heart OAR; no chamber prediction",
        "annotation_status":"draft_prediction","expert_approval":False,"clinical_status":"unknown",
        "source_DICOM_unchanged":unchanged,"original_grid_shape":list(mask.shape),
        "spacing_mm":nib.affines.voxel_sizes(image.affine).tolist(),"predicted_voxels":int(mask.sum()),
        "empty_prediction":not bool(mask.any()),"elapsed_seconds":time.perf_counter()-start,"device":a.device,
        "mesh":metrics,"checkpoint_SHA256":sha256_file(a.checkpoint),"local_reference_accuracy_available":False}
    (output/"pilot_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print("PILOT",json.dumps({k:summary[k] for k in ["series_alias","scope","predicted_voxels","elapsed_seconds","source_DICOM_unchanged"]}),flush=True)
    return summary

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-root",type=Path,required=True);p.add_argument("--series",required=True)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    p.add_argument("--reserve-GB",type=float,default=80);pilot(p.parse_args())
