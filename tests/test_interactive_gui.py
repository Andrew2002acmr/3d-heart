"""Small local Qt/VTK smoke test; requires a working graphics context."""
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6 import QtWidgets

from heart3d.interactive.data import CaseSource, LoadedCase
from heart3d.interactive.state import plane_corners
from heart3d.interactive.window import ViewerWindow
from heart3d.labels import LABELS
from heart3d.volume import Case


def test_widgets_update_slices_actors_and_planes():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    mask = np.zeros((12, 10, 8), dtype=np.uint8)
    mask[2:6, 2:6, 2:6] = 1
    volume = Case(mask.astype(float)*20, mask, np.eye(4), "unknown", {})
    source = CaseSource("synthetic", Path("ct"), Path("mask"), [], [])
    original = pv.Sphere(center=(4, 4, 4), radius=2)
    loaded = LoadedCase(source, volume, {1: {"Original": original, "Smoothed": original.copy()}}, [], None, (0, 40))
    window = ViewerWindow([source], autoload=False)
    try:
        window.accept_case(loaded)
        window.show()
        app.processEvents()
        assert not window.structure_checks[5].isEnabled()
        assert "MYO" in window.warnings.toPlainText()
        assert window.state.variants[1] == "Original"
        assert len(window.scene.plotter.renderer.lights) > 0
        panel = window.slices[2]
        panel.slider.setValue(3)
        panel.alpha.setValue(100)
        window.plane_toggle.setChecked(True)
        window.refresh()
        assert panel.index.value() == window.state.indices[2] == 3
        assert panel.canvas.image.pixelColor(3, 10-1-3).name() == LABELS[1][2]
        np.testing.assert_array_equal(window.scene.planes[2][0].points, plane_corners(volume, 2, 3))
        assert window.scene.planes[2][1].GetVisibility()
        panel.mask.setChecked(False)
        window.refresh()
        color = panel.canvas.image.pixelColor(3, 6)
        assert color.red() == color.green() == color.blue()
        assert window.state.plane_overlays[1]  # Other views stay enabled.
        panel.mask.setChecked(True)
        window.mask_all.setChecked(False)
        assert not window.state.overlay
        window.mask_all.setChecked(True)
        window.structure_checks[1].setChecked(False)
        window.refresh()
        assert not window.scene.actors[1].GetVisibility()
        color = panel.canvas.image.pixelColor(3, 6)
        assert color.red() == color.green() == color.blue()
        window.structure_checks[1].setChecked(True)
        window.opacity_controls[1].setValue(35)
        window.variant_selectors[1].setCurrentText("Smoothed")
        window.refresh()
        assert window.scene.displayed[1] == "Smoothed"
        assert window.scene.actors[1].GetProperty().GetOpacity() == .35
        np.testing.assert_allclose(window.scene.actors[1].prop.color.float_rgb, pv.Color(LABELS[1][2]).float_rgb)
        image = window.scene.plotter.screenshot(return_img=True)
        assert image.std() > 1
        window.scene.views.setCurrentIndex(3)
        window.scene.reset_button.click()
        assert window.scene.views.currentIndex() == 0
        window.scene.set_state(None)
        window.scene.set_state(window.state)
        assert len(window.scene.plotter.renderer.lights) > 0
    finally:
        window.close()
        app.processEvents()
