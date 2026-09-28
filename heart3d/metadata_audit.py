"""Read-only audit of ImageCHD XLSX keys. Never assigns patient geometry."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from .volume import sha256

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def first_sheet_cells(path):
    """Read values from the first sheet of the dataset's static XLSX tables.

    No formulas are evaluated, no workbook is modified. Reject formulas rather
    than treating cached values as authoritative geometry metadata.
    """
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheet = workbook.find("s:sheets/s:sheet", NS)
        rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        target = next(r.attrib["Target"] for r in rels if r.attrib["Id"] == rid)
        member = target.lstrip("/") if target.startswith("/") else "xl/" + target
        root = ET.fromstring(z.read(member))
        rows = []
        for row in root.findall("s:sheetData/s:row", NS):
            cells = {}
            for cell in row.findall("s:c", NS):
                if cell.find("s:f", NS) is not None:
                    raise ValueError(f"Formula in {path.name}:{cell.attrib['r']}; manual review needed")
                value = cell.find("s:v", NS)
                text = value.text if value is not None else None
                kind = cell.attrib.get("t")
                if kind == "s" and text is not None:
                    text = strings[int(text)]
                elif kind == "inlineStr":
                    text = "".join(cell.find("s:is", NS).itertext())
                elif text is not None and kind not in ("str", "e"):
                    text = float(text)
                cells[re.sub(r"\d+", "", cell.attrib["r"])] = text
            rows.append((int(row.attrib["r"]), cells))
    return sheet.attrib["name"], rows


def compare_keys(file_ids, metadata_ids, classification_ids):
    duplicates = lambda values: sorted(k for k, n in Counter(values).items() if n > 1)
    f, m, c = set(file_ids), set(metadata_ids), set(classification_ids)
    offsets = {a - b for a, b in zip(sorted(f), sorted(m))}
    offset = next(iter(offsets)) if len(f) == len(m) and len(offsets) == 1 else None
    return {
        "file_count": len(file_ids), "metadata_count": len(metadata_ids),
        "classification_count": len(classification_ids),
        "duplicates": {"files": duplicates(file_ids), "metadata": duplicates(metadata_ids),
                       "classification": duplicates(classification_ids)},
        "classification_matches_file_ids": f == c,
        "constant_offset_matching_all_keys": offset,
        "candidate_rule": None if offset is None else f"file_id = idx + {offset}",
        "patient_row_correspondence_confirmed": False,
        "status": "structural_key_match_only; no author crosswalk or original series identifiers",
        "apply_metadata_spacing": False,
    }


def audit_metadata(data):
    inventory = json.loads((data / "archive_inventory.json").read_text(encoding="utf-8"))
    file_ids = [int(re.search(r"ct_(\d+)_image", x["name"])[1]) for x in inventory if x["name"].endswith("_image.nii.gz")]
    info_path = data / "imagechd_dataset_image_info.xlsx"
    diagnosis_path = data / "imageCHD_dataset_info.xlsx"
    sheet, raw_rows = first_sheet_cells(info_path)
    geometry_rows = []
    for row, values in raw_rows:
        if str(values.get("A", "")).strip() != "idx":
            raise ValueError(f"Unexpected metadata schema in row {row}")
        if str(values.get("I", "")).strip() != "PixelSpacing" or str(values.get("L", "")).strip() != "calculate_z_thick":
            raise ValueError(f"Unexpected geometry columns in row {row}")
        geometry_rows.append({"row": row, "idx": int(values["B"]),
                              "pixel_spacing": [values["J"], values["K"]],
                              "calculate_z_thick": values["M"],
                              "cells": f"B{row},J{row}:K{row},M{row}"})
    _, diagnosis_rows = first_sheet_cells(diagnosis_path)
    classification_ids = [int(values["A"]) for _, values in diagnosis_rows if isinstance(values.get("A"), (int, float))]
    keys = compare_keys(file_ids, [r["idx"] for r in geometry_rows], classification_ids)
    # Only ID and geometry columns leave the input workbook; no dates/sex/diagnoses.
    return {"keys": keys, "geometry_sheet": sheet, "geometry_rows": geometry_rows,
            "source_sha256": {p.name: sha256(p) for p in (info_path, diagnosis_path, data / "archive_inventory.json")},
            "z_semantics_confirmed": False,
            "note": "calculate_z_thick is an author field, not proof of exported slice-centre spacing. No DICOM IOP/IPP or conversion log supplied."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_metadata(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["keys"], indent=2))
