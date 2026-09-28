"""Small native PyVista viewer. No web app or server."""
import json
from pathlib import Path
import pyvista as pv

from .labels import LABELS


def make_viewer(meshes, unit, off_screen=False):
    plotter = pv.Plotter(title="ImageCHD | existing segmentation", window_size=(1280, 900), off_screen=off_screen)
    plotter.set_background("#17212b")
    actors, callbacks = {}, {}
    for row, (label, mesh) in enumerate(sorted(meshes.items())):
        short, name, color = LABELS[label]
        actor = plotter.add_mesh(mesh, color=color, smooth_shading=False, name=short)
        actors[label] = actor
        def toggle(visible, target=actor):
            target.SetVisibility(bool(visible))
            plotter.render()
        callbacks[label] = toggle
        y = 140 + row * 38
        plotter.add_checkbox_button_widget(toggle, value=True, position=(18, y), size=24,
                                          color_on=color, color_off="#444b55", background_color="#17212b")
        plotter.add_text(f"{label}  {short}  {name}", position=(53, y + 2), font_size=11, color="white")
        # Number keys and checkboxes use the same visibility state.
        widget = plotter.widgets.button_widgets[-1]
        def key_toggle(target=actor, button=widget, callback=toggle):
            state = not target.GetVisibility()
            button.GetRepresentation().SetState(int(state))
            callback(state)
        plotter.add_key_event(str(label), key_toggle)
    unit_text = "Coordinates: mm (NIfTI header)" if unit == "mm" else "Scale unknown: header units, NOT verified mm"
    plotter.add_text("ImageCHD - ready-made segmentation\n" + unit_text + "\nAxes from header; anatomical orientation unverified",
                     position="upper_left", font_size=12, color="white")
    plotter.add_text("Drag: rotate. Wheel: zoom. Middle drag: pan.\n"
                     "Checkboxes / 1-7: visibility. r: reset. q: close.",
                     position="lower_left", font_size=11, color="#d6dde5")
    plotter.add_axes(xlabel="R", ylabel="A", zlabel="S", color="white")
    plotter.view_isometric()
    plotter.reset_camera()
    return plotter, actors, callbacks


def view_directory(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    meshes = {s["label"]: pv.read(directory / s["file"]) for s in report["structures"]}
    plotter, _, _ = make_viewer(meshes, report["geometry"]["unit"])
    plotter.show()
