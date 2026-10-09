import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description="ImageCHD: CT + existing mask -> meshes -> 3D")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="List paired NIfTI files recursively")
    scan.add_argument("directory", type=Path)
    build = commands.add_parser("build", help="Validate a pair, save overlays, VTP meshes and report")
    build.add_argument("--ct", required=True, type=Path)
    build.add_argument("--mask", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--window", nargs=2, type=float, metavar=("MIN", "MAX"))
    build.add_argument("--preview", action="store_true", help="Save off-screen 3D PNG (requires VTK rendering)")
    view = commands.add_parser("view", help="Open exported meshes in interactive PyVista window")
    view.add_argument("directory", type=Path)
    gui = commands.add_parser("gui", help="Open CT, segmentation and saved VTP in a local Qt window")
    gui.add_argument("--data", type=Path, default=Path("data/imagechd"))
    gui.add_argument("--results", type=Path, default=Path("outputs"))
    gui.add_argument("--case", dest="case_id", default=None)
    edit = commands.add_parser("edit", help="Original-grid research mask editor with linked CT/3D")
    for name in ("ct", "mask", "geometry", "output"):
        edit.add_argument("--"+name, type=Path, required=True)
    edit.add_argument("--schema", choices=("heart", "cardiac7"), default="heart")
    args = parser.parse_args()
    try:
        if args.command == "scan":
            from .volume import discover
            print(json.dumps(discover(args.directory), indent=2, ensure_ascii=False))
        elif args.command == "gui":
            try:
                from .interactive.window import run_app
            except ModuleNotFoundError as error:
                parser.exit(2, f"GUI dependency missing: {error}. Install requirements-lock.txt.\n")
            return run_app(args.data, args.results, args.case_id)
        elif args.command == "edit":
            from .interactive.editor import run_editor
            return run_editor(args.ct, args.mask, args.geometry, args.output, args.schema)
        elif args.command == "view":
            from .viewer import view_directory
            view_directory(args.directory)
        else:
            from .volume import load_case
            from .surfaces import export_surfaces
            from .slices import save_slices
            if args.out.exists() and any(args.out.iterdir()):
                raise ValueError("Output directory is not empty. Choose a new --out to preserve previous results.")
            print("Loading CT/mask and checking geometry...", flush=True)
            case = load_case(args.ct, args.mask)
            args.out.mkdir(parents=True, exist_ok=True)
            save_slices(case, args.out / "slices.png", args.window)
            print("Building surfaces at full voxel resolution...", flush=True)
            meshes = export_surfaces(case, args.out / "meshes")
            case.report["created_utc"] = datetime.now(timezone.utc).isoformat()
            case.report["software"] = {name: version(name) for name in
                ("numpy", "nibabel", "scipy", "scikit-image", "matplotlib", "pyvista", "vtk")}
            case.report["software"]["python"] = sys.version
            # Save the scientific result before optional graphics initialization.
            report_path = args.out / "report.json"
            report_path.write_text(json.dumps(case.report, indent=2, ensure_ascii=False), encoding="utf-8")
            if args.preview:
                from .viewer import make_viewer
                plotter, _, _ = make_viewer(meshes, case.unit, off_screen=True)
                plotter.show(screenshot=str(args.out / "preview.png"))
            print(f"Saved: {args.out.resolve()}")
            for warning in case.report["warnings"]:
                print(f"WARNING: {warning}")
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
