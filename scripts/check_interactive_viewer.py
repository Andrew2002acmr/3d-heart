"""Local acceptance check of the actual Qt window using existing case outputs.

No reconstruction or postprocessing. Screenshots and JSON stay under --out.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6 import QtTest, QtWidgets

from heart3d.interactive.data import catalog
from heart3d.interactive.state import plane_corners
from heart3d.interactive.window import ViewerWindow, configure_app
from heart3d.labels import LABELS
from heart3d.volume import sha256


def run(data, results, cases, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new, empty --out directory")
    sources, notes = catalog(data, results)
    selected = cases or [s.case_id for s in sources]
    indices = {s.case_id: i for i, s in enumerate(sources)}
    if not selected or any(c not in indices for c in selected):
        raise ValueError("No matching cases; check --data / --cases")
    out.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_app(app)
    window = ViewerWindow(sources, notes, autoload=False)
    screen = app.primaryScreen().availableGeometry()
    window.resize(min(1440, screen.width()-40), min(940, screen.height()-60))
    window.show()
    errors, records = [], []
    window.load_failed.connect(errors.append)
    try:
        for case_id in selected:
            print(f"{case_id}: loading saved data...", flush=True)
            started = time.perf_counter()
            i = indices[case_id]
            if i == window.case_selector.currentIndex():
                window.start_load(i)
            else:
                window.case_selector.setCurrentIndex(i)
            while window.worker is not None:
                app.processEvents()
                # Release Python's GIL for the loading worker; QTest.qWait may
                # otherwise starve Python work and distort the measured time.
                time.sleep(.02)
                if time.perf_counter()-started > 180:
                    raise TimeoutError(f"Loading {case_id} exceeded 180 seconds")
            if errors:
                raise RuntimeError(errors[-1])
            state = window.state
            assert state is not None and state.loaded.source.case_id == case_id
            assert all("Original" in v for v in state.loaded.meshes.values()), "Missing saved 3D surfaces"
            seconds = time.perf_counter()-started
            print(f"{case_id}: loaded in {seconds:.2f}s; checking controls...", flush=True)
            assert all(v == "Original" for v in state.variants.values())
            assert set(window.scene.actors) == {n for n, v in state.loaded.meshes.items() if v}
            assert "не подтвержд" in window.warnings.toPlainText()
            for n in LABELS:
                assert window.structure_checks[n].isEnabled() == (n in state.visible)
            window.plane_toggle.setChecked(True)
            for axis, panel in window.slices.items():
                for index in (0, state.loaded.volume.ct.shape[axis]-1, state.indices[axis]//2):
                    panel.slider.setValue(index)
                    assert panel.index.value() == state.indices[axis] == index
                panel.alpha.setValue(65)
                panel.mask.setChecked(False)
                assert not state.plane_overlays[axis]
                panel.mask.setChecked(True)
                panel.alpha.setValue(42)
                panel.slider.setValue((state.loaded.volume.ct.shape[axis]-1)//2)
            window.refresh()
            for axis, (mesh, actor) in window.scene.planes.items():
                assert actor.GetVisibility()
                np.testing.assert_array_equal(mesh.points, plane_corners(state.loaded.volume, axis, state.indices[axis]))
            window.mask_all.setChecked(False)
            assert not state.overlay
            window.mask_all.setChecked(True)
            for label in sorted(state.visible):
                window.structure_checks[label].setChecked(False)
                window.refresh()
                if label in window.scene.actors:
                    assert not window.scene.actors[label].GetVisibility()
                window.structure_checks[label].setChecked(True)
                window.opacity_controls[label].setValue(40)
                selector = window.variant_selectors[label]
                for j in range(selector.count()):
                    selector.setCurrentIndex(j)
                    window.refresh()
                    if label in window.scene.actors:
                        assert window.scene.displayed[label] == selector.currentText()
                        assert window.scene.actors[label].GetProperty().GetOpacity() == .4
                selector.setCurrentText("Original")
                window.opacity_controls[label].setValue(100)
            camera = window.scene.plotter.camera
            before = np.array(camera.position)
            camera.azimuth += 30
            assert not np.allclose(before, camera.position)
            angle = camera.view_angle
            camera.zoom(1.2)
            assert camera.view_angle < angle
            # Pan translates camera and focal point together without changing direction.
            delta = np.array([10., 0., 0.])
            direction = np.array(camera.focal_point)-camera.position
            camera.position = np.array(camera.position)+delta
            camera.focal_point = np.array(camera.focal_point)+delta
            np.testing.assert_allclose(np.array(camera.focal_point)-camera.position, direction)
            for view in range(window.scene.views.count()):
                window.scene.views.setCurrentIndex(view)
            window.scene.reset_button.click()
            window.plane_toggle.setChecked(False)
            window.refresh()
            QtTest.QTest.qWait(150)
            assert window.grab().save(str(out / f"{case_id}_window.png"))
            render = window.scene.plotter.screenshot(str(out / f"{case_id}_3d.png"), return_img=True)
            assert render.std() > 1
            unchanged = all(sha256(path) == state.loaded.volume.report[key]["sha256"]
                            for key, path in (("ct", state.loaded.source.ct), ("mask", state.loaded.source.mask)))
            assert unchanged
            record = {"case": case_id, "shape": list(state.loaded.volume.ct.shape),
                      "labels": sorted(state.visible), "load_seconds": round(seconds, 2),
                      "report": str(state.loaded.report_path),
                      "variants": {LABELS[n][0]: list(v) for n, v in state.loaded.meshes.items()},
                      "warnings": window.warnings.toPlainText(), "inputs_unchanged": unchanged,
                      "widget_camera_and_plane_checks_passed": True}
            records.append(record)
            (out / "check.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{case_id}: OK, {len(state.visible)} structures, load {seconds:.2f}s", flush=True)
            # Do not retain previous large arrays while the next worker is loading.
            del state
    finally:
        window.close()
        while window.worker is not None:
            app.processEvents()
            time.sleep(.02)
        app.processEvents()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/imagechd"))
    parser.add_argument("--results", type=Path, default=Path("outputs"))
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.results, args.cases, args.out)
