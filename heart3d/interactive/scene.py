"""Embedded PyVista scene, reading meshes prepared by the existing pipeline."""
import os
os.environ["QT_API"] = "pyside6"
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets

from ..labels import LABELS
from .state import plane_corners


class ScenePanel(QtWidgets.QGroupBox):
    def __init__(self, off_screen=False):
        super().__init__("3D · сохранённые поверхности")
        self.state = None
        self.actors, self.displayed, self.planes = {}, {}, {}
        self.plotter = QtInteractor(self, off_screen=off_screen, auto_update=False, multi_samples=0)
        self.plotter.set_background("#17212b")
        if self.plotter.iren is not None:
            self.plotter.enable_trackball_style()
        self.plotter.enable_depth_peeling(number_of_peels=4)
        self.reset_button = QtWidgets.QPushButton("Сброс камеры")
        self.views = QtWidgets.QComboBox()
        self.views.addItems(["Изометрия", "+X (R*)", "−X (L*)", "+Y (A*)", "−Y (P*)", "+Z (S*)", "−Z (I*)"])
        self.views.setToolTip("* Направления только по заголовку; ориентация пациента не подтверждена")
        self.reset_button.clicked.connect(self.reset_camera)
        self.views.currentIndexChanged.connect(self.set_view)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(self.views)
        bar.addWidget(self.reset_button)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.plotter.interactor, 1)
        layout.addLayout(bar)
        hint = QtWidgets.QLabel("ЛКМ: вращение · колесо: zoom · средняя кнопка / Shift+ЛКМ: pan")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #8797a5")
        layout.addWidget(hint)

    def set_state(self, state):
        self.plotter.clear()
        self.state = state
        self.actors, self.displayed, self.planes = {}, {}, {}
        if state is not None:
            for label, variants in state.loaded.meshes.items():
                if "Original" in variants:
                    self._mesh(label)
            for axis, color in enumerate(("#df9360", "#6ac5b4", "#7dacf2")):
                mesh = pv.PolyData(plane_corners(state.loaded.volume, axis, state.indices[axis]), [4, 0, 1, 2, 3])
                actor = self.plotter.add_mesh(mesh, color=color, opacity=.12, show_edges=True,
                                               edge_color=color, name=f"slice-plane-{axis}", render=False,
                                               reset_camera=False, pickable=False)
                actor.SetUseBounds(False)
                actor.SetVisibility(state.planes)
                self.planes[axis] = (mesh, actor)
            self.plotter.add_axes(xlabel="X*", ylabel="Y*", zlabel="Z*")
            self.reset_camera()
        self.plotter.render()

    def _mesh(self, label):
        variant = self.state.variants[label]
        mesh = self.state.loaded.meshes[label][variant]
        actor = self.plotter.add_mesh(mesh, color=LABELS[label][2], smooth_shading=False,
                                      opacity=self.state.opacity[label], name=f"structure-{label}",
                                      reset_camera=False, render=False)
        actor.SetVisibility(label in self.state.visible)
        self.actors[label], self.displayed[label] = actor, variant

    def refresh(self):
        if self.state is None:
            return
        for label in self.actors:
            if self.displayed[label] != self.state.variants[label]:
                self._mesh(label)
            self.actors[label].SetVisibility(label in self.state.visible)
            self.actors[label].GetProperty().SetOpacity(self.state.opacity[label])
        for axis, (mesh, actor) in self.planes.items():
            mesh.points = plane_corners(self.state.loaded.volume, axis, self.state.indices[axis])
            actor.SetVisibility(self.state.planes)
        self.plotter.render()

    def reset_camera(self):
        self.views.blockSignals(True)
        self.views.setCurrentIndex(0)
        self.views.blockSignals(False)
        self.plotter.view_isometric()
        self.plotter.reset_camera()

    def set_view(self, index):
        if index == 0:
            self.plotter.view_isometric()
        else:
            vectors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
            self.plotter.view_vector(vectors[index-1], viewup=(0,1,0) if index >= 5 else (0,0,1))
        self.plotter.reset_camera()

    def shutdown(self):
        self.plotter.close()
