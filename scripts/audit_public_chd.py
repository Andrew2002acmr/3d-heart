"""Audit all downloaded CT/label pairs and conservatively group release overlap."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import urllib.request
from heart3d.volume import load_case, sha256
from scripts.audit_pediatric_overlap import inventory
SOURCE = "https://raw.githubusercontent.com/XiaoweiXu/Whole-heart-and-great-vessel-segmentation-of-chd_segmentation/master/README.md"
CODES = ["ASD","AVSD","VSD","AD","ToF","PAS","PDA","CA","CAT","PS","AAA","TGA","SV","PuA","Normal"]

def diagnoses(text):
    rows = {}
    for line in text.splitlines():
        fields = re.split(r"[\s|]+", line.strip().strip("|").strip())
        if len(fields) == 16 and fields[0].isdigit() and all(v in ("0","1") for v in fields[1:]):
            rows["ct_"+fields[0]] = [k for k,v in zip(CODES,fields[1:]) if v == "1"]
    return rows

def audit(root, image_root, output, summary_path=None):
    root, image_root, output = Path(root), Path(image_root), Path(output)
    output.mkdir(parents=True,exist_ok=True)
    source_path = output/"source_readme.txt"
    if not source_path.exists():
        with urllib.request.urlopen(SOURCE,timeout=60) as response:
            source_path.write_bytes(response.read())
    diagnosis = diagnoses(source_path.read_text(encoding="utf-8"))
    prior = inventory(image_root/"archive_inventory_previous.json")
    rows = []
    for ct in sorted((root/"pairs").glob("*_image.nii.gz")):
        case = ct.name.removesuffix("_image.nii.gz")
        mask = ct.with_name(case+"_label.nii.gz")
        row = {"case_id":case,"source":"CHD68","modality":"CT","age":None,"age_verified":False,
               "diagnosis_codes":diagnosis.get(case,[]),"diagnosis_source":SOURCE,
               "patient_linkage_independently_verified":False,"duplicate_group":"chd68_imagechd::"+case,
               "split_eligible_0_17":False,"physical_context_eligible":False}
        codes = row["diagnosis_codes"]
        row["healthy_status"] = ("pathological" if set(codes)-{"Normal"} else
            "no_structural_cardiac_abnormality_reported" if "Normal" in codes else "unknown")
        try:
            volume = load_case(ct,mask)
            row.update({"pair_grid_gate":"passed","image_SHA256":volume.report["ct"]["sha256"],
                "mask_SHA256":volume.report["mask"]["sha256"],"shape":list(volume.ct.shape),
                "spacing":volume.report["geometry"]["spacing"],"spatial_unit":volume.unit,
                "physical_scale_verified":False,"label_counts":volume.report["label_counts"],
                "noncardiac_labels":volume.report["ignored_labels"],
                "missing_cardiac_labels":volume.report["missing_labels"],
                "intensity_min_max":[volume.report["intensity"]["min"],volume.report["intensity"]["max"]],
                "HU_independently_verified":False})
            del volume
            matching, sha_matches = [], []
            for filename, hashkey in [(ct.name,"image_SHA256"),(mask.name,"mask_SHA256")]:
                if filename in prior:matching.append(filename)
                counterpart = image_root/"pairs"/filename
                if counterpart.exists() and sha256(counterpart)==row[hashkey]:sha_matches.append(filename)
            row["imagechd_same_name_in_archive"] = matching
            row["imagechd_SHA256_identical_available_files"] = sha_matches
            row["imagechd_SHA256_complete_pair"] = len(sha_matches)==2
            row["readiness"] = "requires_age_and_physical_scale_review"
        except (OSError, ValueError) as error:
            row.update({"pair_grid_gate":"failed","readiness":"pair_failure","reason":str(error)})
        rows.append(row)
        print("AUDITED",case,row["pair_grid_gate"],flush=True)
    report = {"version":1,"dataset":"CHD68","cases":rows,"pair_count":len(rows),
        "pair_gate_passed":sum(r["pair_grid_gate"]=="passed" for r in rows),
        "complete_pair_SHA_overlap_verified":sum(r.get("imagechd_SHA256_complete_pair",False) for r in rows),
        "declared_units":dict(Counter(r.get("spatial_unit","failed") for r in rows)),
        "diagnosis_source_SHA256":sha256(source_path),"diagnosis_rows_found":len(diagnosis),
        "training_started":False,"frozen_split_created":False,
        "blocking_reasons":["Individual ages unavailable","Physical scale not independently verified"],
        "deduplication_policy":"Group same release case and all CRC/name overlap before any split; available pair SHA checks do not establish all-patient linkage.",
        "source_readme":SOURCE}
    (output/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    if summary_path:Path(summary_path).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("SUMMARY",json.dumps({k:report[k] for k in ("pair_count","pair_gate_passed","complete_pair_SHA_overlap_verified","declared_units","diagnosis_rows_found")}),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chd68-root",type=Path,required=True);p.add_argument("--imagechd-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--summary",type=Path)
    a=p.parse_args();audit(a.chd68_root,a.imagechd_root,a.output,a.summary)
