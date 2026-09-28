import numpy as np
from heart3d.surfaces import build_surface
from heart3d.mesh_quality import quality


def test_marching_cubes_winding_not_changed_at_digital_contacts():
    # Enumerate all 256 binary configurations of a 2x2x2 cell, with padding.
    # This catches orientation regressions around ambiguous digital contacts.
    for bits in range(1, 256):
        mask = np.array([(bits >> i) & 1 for i in range(8)], dtype=np.uint8).reshape(2, 2, 2)
        mesh, _ = build_surface(mask, 1, np.eye(4))
        assert quality(mesh)["inconsistent_winding_edges"] == 0, bits
