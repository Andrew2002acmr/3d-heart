"""Reproducible multi-case audit. Original NIfTI files are read-only."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

from .geometry_audit import audit_headers
from .metadata_audit import audit_metadata
from .volume import load_case, sha256
from .slices import save_slices
from .surfaces import export_surfaces


def run(data, cases, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new, empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
    metadata = audit_metadata(data)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    headers = audit_headers(data, cases)
    (out / "headers.json").write_text(json.dumps(headers, indent=2) + "\n", encoding="utf-8")
    summary = {"cases": [], "source_code_sha256": source_hashes,
               "versions": {name: version(name) for name in ("numpy", "nibabel", "scipy", "scikit-image", "pyvista", "vtk", "SimpleITK")},
               "geometry_policy": "Original header units; no metadata spacing overrides or anatomical flips"}
    for case_id in cases:
        print(f"{case_id}: load and validate", flush=True)
        ct_path, mask_path = (data / f"{case_id}_{suffix}.nii.gz" for suffix in ("image", "label"))
        before = {p.name: sha256(p) for p in (ct_path, mask_path)}
        case = load_case(ct_path, mask_path)
        directory = out / case_id
        directory.mkdir()
        save_slices(case, directory / "slices.png")
        meshes = export_surfaces(case, directory / "meshes")
        for name in ("ct", "mask"):
            case.report[name]["path"] = Path(case.report[name]["path"]).name
        after = {p.name: sha256(p) for p in (ct_path, mask_path)}
        if before != after:
            raise RuntimeError("Input changed during audit")
        case.report["input_hashes_unchanged"] = True
        case.report["software"] = summary["versions"]
        (directory / "report.json").write_text(json.dumps(case.report, indent=2) + "\n", encoding="utf-8")
        summary["cases"].append({"case": case_id, "shape": list(case.ct.shape),
                                 "labels": list(case.report["label_counts"]),
                                 "geometry": case.report["geometry"],
                                 "structures": case.report["structures"],
                                 "warnings": case.report["warnings"],
                                 "input_hashes_unchanged": True})
        (out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        del meshes, case
        print(f"{case_id}: done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.cases, args.out)
