"""Parse release metadata without exporting dates; index linkage stays provisional."""
from datetime import datetime
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from heart3d.storage import sha256_file
from scripts.audit_pediatric_overlap import inventory
from scripts.audit_public_chd import diagnoses

NS={"m":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
def cells(path):
    with zipfile.ZipFile(path) as archive:
        strings=[]
        if "xl/sharedStrings.xml" in archive.namelist():
            for si in ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("m:si",NS):
                strings.append("".join(n.text or "" for n in si.iter("{"+NS["m"]+"}t")))
        rows=[]
        for row in ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall(".//m:row",NS):
            result={}
            for cell in row.findall("m:c",NS):
                value=cell.find("m:v",NS);text=value.text if value is not None else ""
                if cell.get("t")=="s":text=strings[int(text)]
                if cell.get("t")=="inlineStr":text="".join(n.text or "" for n in cell.findall(".//m:t",NS))
                column="".join(c for c in cell.get("r","") if c.isalpha());result[column]=text.strip()
            rows.append(result)
        return rows

def candidates(path):
    result={}
    for row in cells(path):
        if row.get("A")!="idx" or row.get("C")!="PatientBirthDate" or row.get("E")!="AcquisitionDate":
            raise ValueError("Metadata schema differs")
        index=int(float(row["B"]));case="ct_"+str(1000+index)
        if case in result:
            raise ValueError("Duplicate metadata index requires review")
        item={"metadata_index":index,"release_case_candidate":case,"linkage_rule":"idx + 1000",
              "linkage_author_confirmed":False,"age_verified_for_release":False}
        try:
            birth=datetime.strptime(str(int(float(row["D"]))),"%Y%m%d")
            acquired=datetime.strptime(str(int(float(row["F"]))),"%Y%m%d")
            days=(acquired-birth).days
            if days<0:raise ValueError("Negative age")
            years=acquired.year-birth.year-((acquired.month,acquired.day)<(birth.month,birth.day))
            spacing=[float(row["J"]),float(row["K"])]
            z=float(row["M"])
            if any(not math.isfinite(v) or v<=0 for v in [*spacing,z]):
                raise ValueError("Invalid source spacing candidate")
            item.update({"age_candidate_days":days,"age_candidate_completed_years":years,
                         "candidate_in_0_17":0<=years<=17,
                         "source_pixel_spacing_candidate_mm":spacing,
                         "source_calculated_z_candidate_mm":z,"scanner_model":row["O"]})
        except (ValueError,KeyError):
            item["metadata_values_status"]="requires_review"
        result[case]=item
    return result

def enrich(image_root,audit_path,output,summary):
    image_root,audit_path,output=Path(image_root),Path(audit_path),Path(output)
    table=image_root/"metadata/imagechd_dataset_image_info.xlsx"
    mapped=candidates(table)
    names={n.removesuffix("_image.nii.gz") for n in inventory(image_root/"archive_inventory_previous.json") if n.endswith("_image.nii.gz")}
    report=json.loads(audit_path.read_text())
    codes=diagnoses((audit_path.parent/"source_readme.txt").read_text())
    for row in report["cases"]:
        dx=codes.get(row["case_id"],[])
        row["diagnosis_codes"]=dx
        row["healthy_status"]=("pathological" if set(dx)-{"Normal"} else "no_structural_cardiac_abnormality_reported" if "Normal" in dx else "unknown")
        row["metadata_candidate"]=mapped.get(row["case_id"])
        row["metadata_conflicts_with_published_age_range"]=bool(row["metadata_candidate"] and row["metadata_candidate"].get("age_candidate_completed_years",0)>21)
    report.update({"diagnosis_rows_found":len(codes),"diagnosis_source_SHA256":sha256_file(audit_path.parent/"source_readme.txt"),
        "metadata_candidates_present":sum(r["metadata_candidate"] is not None for r in report["cases"]),
        "metadata_table_sha256":sha256_file(table),"metadata_linkage_rule":"idx + 1000; case-set identity verified; author linkage not confirmed",
        "metadata_case_set_equals_imagechd_release":set(mapped)==names,
        "metadata_published_age_range_conflicts":sum(r["metadata_conflicts_with_published_age_range"] for r in report["cases"]),
        "blocking_reasons":["Per-case metadata linkage to release requires confirmation","NIfTI grid may differ from source spacing; physical scale not verified"]})
    output.mkdir(parents=True,exist_ok=True)
    info={"table_SHA256":sha256_file(table),"metadata_row_count":len(mapped),"case_sets_identical":set(mapped)==names,
          "raw_dates_exported":False,"patient_sex_exported":False,"candidates":mapped}
    (output/"metadata_candidates.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    audit_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    Path(summary).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("METADATA",json.dumps({k:report[k] for k in ["diagnosis_rows_found","metadata_candidates_present","metadata_case_set_equals_imagechd_release"]}),flush=True)
    print("CANDIDATE_AGES",json.dumps({"in_0_17":sum(r["metadata_candidate"].get("candidate_in_0_17",False) for r in report["cases"] if r["metadata_candidate"]),
        "unavailable":sum(r["metadata_candidate"] is None for r in report["cases"]),
        "clinical_status_counts":{s:sum(r["healthy_status"]==s for r in report["cases"]) for s in ["pathological","no_structural_cardiac_abnormality_reported","unknown"]}}),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["imagechd-root","audit","output","summary"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();enrich(a.imagechd_root,a.audit,a.output,a.summary)
