"""Header geometry cross-check using independent NiBabel and SimpleITK readers."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from .volume import image_geometry, sha256


def sitk_geometry(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3:
        raise ValueError("Independent reader expects 3D")
    direction = np.array(reader.GetDirection()).reshape(3, 3)
    lps = np.eye(4)
    lps[:3, :3] = direction @ np.diag(reader.GetSpacing())
    lps[:3, 3] = reader.GetOrigin()
    ras = np.diag([-1., -1., 1., 1.]) @ lps
    return {"size": list(reader.GetSize()), "spacing": list(reader.GetSpacing()),
            "origin_lps": list(reader.GetOrigin()), "direction_lps": direction.tolist(),
            "affine_ras": ras.tolist(),
            "note": "SimpleITK's physical convention does not establish missing source units or patient orientation"}


def audit_header(path):
    warnings = []
    image = nib.load(path)
    header, affine, unit = image_geometry(image, path.name, warnings)
    independent = sitk_geometry(path)
    return {"file": path.name, "sha256": sha256(path), "nibabel": header,
            "simpleitk": independent,
            "readers_agree_shape": list(image.shape) == independent["size"],
            "readers_agree_affine_ras": bool(np.allclose(affine, independent["affine_ras"], rtol=0, atol=1e-4)),
            "unit": unit, "anatomical_orientation_confirmed": False,
            "source_physical_scale_confirmed": False, "warnings": warnings}


def audit_headers(data, cases):
    results = []
    for case in cases:
        ct = audit_header(data / f"{case}_image.nii.gz")
        mask = audit_header(data / f"{case}_label.nii.gz")
        results.append({"case": case, "ct": ct, "mask": mask,
                        "original_grids_identical": ct["nibabel"]["shape"] == mask["nibabel"]["shape"] and
                        bool(np.allclose(ct["nibabel"]["affine"], mask["nibabel"]["affine"], rtol=0, atol=1e-4)),
                        "corrections_applied": []})
    return {"cases": results, "SimpleITK_version": sitk.Version_VersionString(),
            "coordinate_comparison": "SimpleITK LPS -> RAS by diag(-1,-1,1,1); no image reorientation"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_headers(args.data, args.cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.out}")
