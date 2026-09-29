"""Minimal Qt application coordinating 2D, 3D and a single loading worker."""
from pathlib import Path
import sys

from PySide6 import QtCore, QtWidgets

from ..labels import LABELS
from .data import catalog, load_source
from .scene import ScenePanel
from .slices import SlicePanel
from .state import ViewState


class Loader(QtCore.QThread):
    loaded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.source = source

    def run(self):
        try:
            self.loaded.emit(load_source(self.source))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class ViewerWindow(QtWidgets.QMainWindow):
    case_ready = QtCore.Signal(str)
    load_failed = QtCore.Signal(str)

    def __init__(self, sources, notes=(), initial_case=None, autoload=True, off_screen=False):
        super().__init__()
        self.sources, self.notes = sources, notes
        self.state, self.worker, self.closing = None, None, False
        self.setWindowTitle("ImageCHD · КТ → сегментация → 3D")
        self.resize(1440, 940)
        self.setMinimumSize(1050, 700)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(35)
        self.refresh_timer.timeout.connect(self.refresh)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        header = QtWidgets.QHBoxLayout()
        header.addWidget(QtWidgets.QLabel("Исследование"))
        self.case_selector = QtWidgets.QComboBox()
        self.case_selector.setMinimumWidth(150)
        self.case_selector.addItems([s.case_id for s in sources])
        header.addWidget(self.case_selector)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(130)
        self.progress.setRange(0, 0)
        self.progress.hide()
        header.addWidget(self.progress)
        header.addStretch()
        notice = QtWidgets.QLabel("Исследовательский прототип · масштаб и ориентация пациента не подтверждены")
        notice.setWordWrap(True)
        notice.setStyleSheet("color: #ffc875; font-weight: 600")
        header.addWidget(notice)
        layout.addLayout(header)
        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(body, 1)
        self.sidebar = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(self.sidebar)
        side.setContentsMargins(0, 0, 6, 0)
        self.info = QtWidgets.QLabel("Выберите исследование")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-size: 13px; padding: 6px")
        side.addWidget(self.info)
        self.mask_all = QtWidgets.QCheckBox("Вся маска поверх КТ")
        self.mask_all.setChecked(True)
        side.addWidget(self.mask_all)
        self.plane_toggle = QtWidgets.QCheckBox("Плоскости срезов в 3D")
        side.addWidget(self.plane_toggle)
        structures = QtWidgets.QGroupBox("Структуры · видимость в 2D и 3D")
        grid = QtWidgets.QGridLayout(structures)
        for column, title in enumerate(("Метка", "Модель 3D", "3D, %")):
            grid.addWidget(QtWidgets.QLabel(title), 0, column)
        self.structure_checks, self.variant_selectors, self.opacity_controls = {}, {}, {}
        for row, (label, (short, name, color)) in enumerate(LABELS.items(), 1):
            check = QtWidgets.QCheckBox(short)
            check.setStyleSheet(f"color: {color}; font-weight: 600")
            check.setToolTip(name)
            variant = QtWidgets.QComboBox()
            variant.addItem("Original")
            opacity = QtWidgets.QSpinBox()
            opacity.setRange(0, 100)
            opacity.setValue(100)
            opacity.setToolTip("Непрозрачность структуры в 3D; маска в КТ настраивается отдельно")
            grid.addWidget(check, row, 0)
            grid.addWidget(variant, row, 1)
            grid.addWidget(opacity, row, 2)
            self.structure_checks[label] = check
            self.variant_selectors[label] = variant
            self.opacity_controls[label] = opacity
            check.toggled.connect(lambda value, n=label: self.set_visible(n, value))
            variant.currentTextChanged.connect(lambda value, n=label: self.set_variant(n, value))
            opacity.valueChanged.connect(lambda value, n=label: self.set_opacity(n, value))
        side.addWidget(structures)
        window_group = QtWidgets.QGroupBox("Окно КТ · значения файла, не HU")
        window_layout = QtWidgets.QHBoxLayout(window_group)
        self.window_low, self.window_high = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for caption, widget in (("Мин", self.window_low), ("Макс", self.window_high)):
            widget.setDecimals(0)
            widget.setRange(-1e7, 1e7)
            widget.setSingleStep(20)
            window_layout.addWidget(QtWidgets.QLabel(caption))
            window_layout.addWidget(widget)
            widget.valueChanged.connect(self.change_window)
        side.addWidget(window_group)
        side.addWidget(QtWidgets.QLabel("Предупреждения и источник моделей"))
        self.warnings = QtWidgets.QPlainTextEdit()
        self.warnings.setReadOnly(True)
        self.warnings.setMinimumHeight(100)
        side.addWidget(self.warnings, 1)
        self.sidebar.setMinimumWidth(290)
        self.sidebar.setMaximumWidth(365)
        body.addWidget(self.sidebar)
        panels = QtWidgets.QWidget()
        panels_layout = QtWidgets.QGridLayout(panels)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        self.slices = {axis: SlicePanel(axis, name) for axis, name in
                       ((2, "Axial · ось k"), (1, "Coronal · ось j"), (0, "Sagittal · ось i"))}
        for (axis, panel), position in zip(self.slices.items(), ((0,0), (0,1), (1,0))):
            panels_layout.addWidget(panel, *position)
            panel.changed.connect(self.schedule_refresh)
            panel.set_state(None)
        self.scene = ScenePanel(off_screen=off_screen)
        panels_layout.addWidget(self.scene, 1, 1)
        for i in range(2):
            panels_layout.setColumnStretch(i, 1)
            panels_layout.setRowStretch(i, 1)
        body.addWidget(panels)
        body.setSizes([325, 1115])
        self.sidebar.setEnabled(False)
        self.mask_all.toggled.connect(self.toggle_mask)
        self.plane_toggle.toggled.connect(self.toggle_planes)
        self.case_selector.currentIndexChanged.connect(self.start_load)
        if not sources:
            self.info.setText("Не найдены пары КТ и маски")
            self.warnings.setPlainText("Проверьте каталог --data. Нужны пары *_image.nii.gz и *_label.nii.gz.\n" + "\n".join(notes))
        elif autoload:
            index = next((i for i, s in enumerate(sources) if s.case_id == initial_case), 0)
            self.case_selector.blockSignals(True)
            self.case_selector.setCurrentIndex(index)
            self.case_selector.blockSignals(False)
            QtCore.QTimer.singleShot(0, lambda: self.start_load(index))

    def start_load(self, index):
        if index < 0 or self.worker is not None or self.closing:
            return
        self.refresh_timer.stop()
        self.state = None
        for panel in self.slices.values():
            panel.set_state(None)
        self.scene.set_state(None)
        self.sidebar.setEnabled(False)
        self.case_selector.setEnabled(False)
        self.progress.show()
        self.info.setText(f"{self.sources[index].case_id}\nЗагрузка…")
        self.warnings.clear()
        self.statusBar().showMessage("Чтение КТ, маски и готовых VTP. Реконструкция не запускается.")
        self.worker = Loader(self.sources[index], self)
        self.worker.loaded.connect(self.accept_case)
        self.worker.failed.connect(self.fail_load)
        self.worker.finished.connect(self.finish_load)
        self.worker.start()

    def accept_case(self, loaded):
        if self.closing:
            return
        self.state = ViewState(loaded)
        present = set(loaded.meshes)
        for label in LABELS:
            check, selector, opacity = self.structure_checks[label], self.variant_selectors[label], self.opacity_controls[label]
            blockers = [QtCore.QSignalBlocker(w) for w in (check, selector, opacity)]
            check.setEnabled(label in present)
            check.setChecked(label in present)
            check.setText(LABELS[label][0] + (" — нет" if label not in present else ""))
            selector.clear()
            variants = loaded.meshes.get(label, {})
            selector.addItems(list(variants) if variants else ["Нет 3D"])
            selector.setCurrentText("Original")
            selector.setEnabled(len(variants) > 1)
            opacity.setValue(100)
            opacity.setEnabled(bool(variants))
            del blockers
        for widget, value in ((self.mask_all, True), (self.plane_toggle, False)):
            blocker = QtCore.QSignalBlocker(widget)
            widget.setChecked(value)
            del blocker
        blockers = [QtCore.QSignalBlocker(w) for w in (self.window_low, self.window_high)]
        self.window_low.setValue(loaded.window[0])
        self.window_high.setValue(loaded.window[1])
        del blockers
        for panel in self.slices.values():
            panel.set_state(self.state)
        self.scene.set_state(self.state)
        shape = " × ".join(str(n) for n in loaded.volume.ct.shape)
        count3d = sum(bool(v) for v in loaded.meshes.values())
        self.info.setText(f"{loaded.source.case_id}\n{shape} вокселей\nСтруктуры: {len(present)}/7 · поверхности 3D: {count3d}")
        missing = [LABELS[n][0] for n in LABELS if n not in present]
        warnings = ["Физический масштаб не подтверждён.", "Ориентация пациента не подтверждена."]
        if missing:
            warnings.append("Отсутствуют метки: " + ", ".join(missing))
        warnings.extend(loaded.warnings)
        if loaded.report_path:
            warnings.append("Отчёт моделей: " + str(loaded.report_path))
        self.warnings.setPlainText("\n\n".join(warnings))
        self.sidebar.setEnabled(True)
        self.statusBar().showMessage("Готово · Original выбран по умолчанию · координаты по заголовку")
        self.case_ready.emit(loaded.source.case_id)

    def finish_load(self):
        thread = self.worker
        self.worker = None
        if thread:
            thread.deleteLater()
        self.progress.hide()
        self.case_selector.setEnabled(True)
        if self.closing:
            self.close()

    def fail_load(self, message):
        self.warnings.setPlainText("Не удалось загрузить случай:\n" + message)
        self.statusBar().showMessage("Ошибка загрузки. Можно выбрать другой случай.")
        self.load_failed.emit(message)

    def schedule_refresh(self, *_):
        if self.state and not self.refresh_timer.isActive():
            self.refresh_timer.start()

    def refresh(self):
        for panel in self.slices.values():
            panel.refresh()
        self.scene.refresh()

    def set_visible(self, label, value):
        if self.state:
            self.state.set_visible(label, value)
            self.schedule_refresh()

    def set_variant(self, label, value):
        if self.state and value in self.state.loaded.meshes.get(label, {}):
            self.state.select_variant(label, value)
            self.schedule_refresh()

    def set_opacity(self, label, value):
        if self.state and label in self.state.opacity:
            self.state.opacity[label] = value/100
            self.schedule_refresh()

    def toggle_mask(self, value):
        if self.state:
            self.state.overlay = value
            self.schedule_refresh()

    def toggle_planes(self, value):
        if self.state:
            self.state.planes = value
            self.schedule_refresh()

    def change_window(self, *_):
        if self.state and self.window_low.value() < self.window_high.value():
            self.state.window = (self.window_low.value(), self.window_high.value())
            self.schedule_refresh()

    def closeEvent(self, event):
        self.refresh_timer.stop()
        if self.worker is not None:
            self.closing = True
            self.statusBar().showMessage("Закрытие после завершения чтения текущего случая…")
            event.ignore()
            return
        self.scene.shutdown()
        super().closeEvent(event)


def configure_app(app):
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { background: #202b36; color: #e5edf5; font-size: 12px; }
        QGroupBox { border: 1px solid #425362; border-radius: 4px; margin-top: 12px; padding-top: 8px; }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit { background: #14202b; padding: 3px; }
        QPushButton { padding: 5px 8px; background: #344857; border-radius: 3px; }
        QWidget:disabled { color: #73818e; }
    """)


def run_app(data_root, results_root, initial_case=None):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    configure_app(app)
    sources, notes = catalog(Path(data_root), Path(results_root))
    window = ViewerWindow(sources, notes, initial_case)
    size = app.primaryScreen().availableGeometry()
    window.resize(min(1440, size.width()-40), min(940, size.height()-60))
    window.show()
    return app.exec()
