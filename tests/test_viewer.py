import numpy as np
import pyvista as pv

from heart3d.viewer import make_viewer


def test_checkbox_event_and_camera_render():
    plotter, actors, _ = make_viewer({1: pv.Sphere()}, "unknown", off_screen=True)
    try:
        before = plotter.screenshot(return_img=True)
        widget = plotter.widgets.button_widgets[0]
        widget.GetRepresentation().SetState(0)
        widget.InvokeEvent("StateChangedEvent")
        assert not actors[1].GetVisibility()
        hidden = plotter.screenshot(return_img=True)
        assert np.any(before != hidden)
        widget.GetRepresentation().SetState(1)
        widget.InvokeEvent("StateChangedEvent")
        assert actors[1].GetVisibility()
        position = np.array(plotter.camera.position)
        plotter.camera.azimuth += 30
        assert not np.allclose(position, plotter.camera.position)
        angle = plotter.camera.view_angle
        plotter.camera.zoom(1.2)
        assert plotter.camera.view_angle < angle
        plotter.render()
    finally:
        plotter.close()
