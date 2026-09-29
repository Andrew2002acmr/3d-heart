"""Qt-free display state and slice/plane geometry."""
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np

from ..labels import LABELS
from .data import LoadedCase


@dataclass
class ViewState:
    loaded: LoadedCase
    visible: set[int] = field(init=False)
    variants: dict[int, str] = field(init=False)
    opacity: dict[int, float] = field(init=False)
    indices: list[int] = field(init=False)
    overlay: bool = True
    overlay_opacity: float = 0.42
    plane_overlays: list[bool] = field(default_factory=lambda: [True, True, True])
    overlay_opacities: dict[int, float] = field(default_factory=dict)
    planes: bool = False

    def __post_init__(self):
        self.visible = set(self.loaded.meshes)
        self.variants = {label: "Original" for label in self.visible}
        self.opacity = {label: 1.0 for label in self.visible}
        self.indices = [(n - 1) // 2 for n in self.loaded.volume.ct.shape]
        self.window = self.loaded.window

    def set_visible(self, label, visible):
        if label not in self.loaded.meshes:
            return
        self.visible.add(label) if visible else self.visible.discard(label)

    def select_variant(self, label, variant):
        if variant not in self.loaded.meshes.get(label, {}):
            raise ValueError("Unavailable surface variant")
        self.variants[label] = variant


def slice_rgb(state, axis):
    selection = [slice(None)] * 3
    selection[axis] = state.indices[axis]
    # Basic slicing avoids np.take's full-volume copies for strided arrays.
    ct = state.loaded.volume.ct[tuple(selection)].T
    mask = state.loaded.volume.mask[tuple(selection)].T
    low, high = state.window
    gray = np.clip((ct - low) / max(high - low, 1e-6), 0, 1) * 255
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    if state.overlay and state.plane_overlays[axis]:
        alpha = state.overlay_opacities.get(axis, state.overlay_opacity)
        for label in state.visible:
            color = LABELS[label][2].lstrip("#")
            color = np.array([int(color[i:i+2], 16) for i in (0, 2, 4)])
            selected = mask == label
            rgb[selected] = (1-alpha)*rgb[selected] + alpha*color
    return np.ascontiguousarray(np.flipud(rgb).astype(np.uint8))


def plane_corners(volume, axis, index):
    """Voxel-centre plane with bounds at outer voxel faces; apply full affine."""
    other = [i for i in range(3) if i != axis]
    corners = np.zeros((4, 3), dtype=float)
    corners[:, axis] = index
    for i, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        corners[i, other[0]] = volume.ct.shape[other[0]] - .5 if u else -.5
        corners[i, other[1]] = volume.ct.shape[other[1]] - .5 if v else -.5
    return nib.affines.apply_affine(volume.affine, corners)
