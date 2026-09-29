"""Small Qt slice panels; pixels and labels stay in the same voxel grid."""
import nibabel as nib
from PySide6 import QtCore, QtGui, QtWidgets

from .state import slice_rgb


class ImageCanvas(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.image = QtGui.QImage()
        self.aspect = 1.
        self.setMinimumSize(180, 100)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#111820"))
        if self.image.isNull():
            return
        width = min(self.width(), self.height() * self.aspect)
        height = width / self.aspect
        target = QtCore.QRectF((self.width()-width)/2, (self.height()-height)/2, width, height)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)
        painter.drawImage(target, self.image)


class SlicePanel(QtWidgets.QGroupBox):
    changed = QtCore.Signal()

    def __init__(self, axis, name):
        super().__init__(name)
        self.axis, self.state = axis, None
        self.canvas = ImageCanvas()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setObjectName(f"slice_{axis}")
        self.index = QtWidgets.QSpinBox()
        self.index.setPrefix("Срез ")
        self.mask = QtWidgets.QCheckBox("Маска")
        self.mask.setChecked(True)
        self.alpha = QtWidgets.QSpinBox()
        self.alpha.setRange(0, 100)
        self.alpha.setSuffix(" %")
        self.alpha.setValue(42)
        self.alpha.setToolTip("Непрозрачность маски в этой плоскости")
        self.axes = QtWidgets.QLabel()
        self.axes.setStyleSheet("color: #8797a5; font-size: 10px")
        controls = QtWidgets.QHBoxLayout()
        for widget in (self.index, self.slider, self.mask, self.alpha):
            controls.addWidget(widget)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 6)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.axes)
        layout.addLayout(controls)
        self.slider.valueChanged.connect(self.index.setValue)
        self.index.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self._index_changed)
        self.mask.toggled.connect(self._overlay_changed)
        self.alpha.valueChanged.connect(self._overlay_changed)

    def set_state(self, state):
        self.state = state
        self.setEnabled(state is not None)
        if state is None:
            self.canvas.image = QtGui.QImage()
            self.canvas.update()
            return
        n = state.loaded.volume.ct.shape[self.axis]
        blockers = [QtCore.QSignalBlocker(w) for w in (self.slider, self.index, self.mask, self.alpha)]
        for widget in (self.slider, self.index):
            widget.setRange(0, n-1)
            widget.setValue(state.indices[self.axis])
        self.index.setSuffix(f" / {n-1}")
        self.mask.setChecked(state.plane_overlays[self.axis])
        self.alpha.setValue(round(100*state.overlay_opacities.get(self.axis, state.overlay_opacity)))
        del blockers
        other = [i for i in range(3) if i != self.axis]
        spacing = nib.affines.voxel_sizes(state.loaded.volume.affine)
        shape = state.loaded.volume.ct.shape
        self.canvas.aspect = shape[other[0]]*spacing[other[0]] / (shape[other[1]]*spacing[other[1]])
        axes = "ijk"
        self.axes.setText(f"Индексы вокселей: {axes[other[0]]} →, {axes[other[1]]} ↑ · направления по заголовку")
        self.refresh()

    def _index_changed(self, index):
        if self.state:
            self.state.indices[self.axis] = index
            self.changed.emit()

    def _overlay_changed(self, *_):
        if self.state:
            self.state.plane_overlays[self.axis] = self.mask.isChecked()
            self.state.overlay_opacities[self.axis] = self.alpha.value()/100
            self.changed.emit()

    def refresh(self):
        if self.state is None:
            return
        rgb = slice_rgb(self.state, self.axis)
        height, width = rgb.shape[:2]
        # Own the copied buffer so Qt never holds a dangling NumPy view.
        self.canvas.image = QtGui.QImage(rgb.data, width, height, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
        self.canvas.update()
