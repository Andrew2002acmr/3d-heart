import json
from pathlib import Path
import numpy as np
from heart3d.ml.single_slice_assist import refine_plane, select_slice


PROTOCOL = json.loads((Path(__file__).parents[1]/"configs/single_slice_assist_v1.json").read_text())


def test_refinement_keeps_strong_agreement_and_never_adds_unsupported_structure():
    p = np.zeros((8, 12, 14), np.float32)
    p[0] = .05; p[1] = .28; p[2] = .28; p[3] = .39
    first = np.ones((12, 14), np.uint8)
    second = np.full((12, 14), 2, np.uint8)
    first[5, 6] = second[5, 6] = 1
    p[:, 5, 6] = 0; p[1, 5, 6] = 1
    result, uncertain, qa = refine_plane(p, np.zeros(first.shape), first, second, PROTOCOL)
    assert result[5, 6] == 1
    assert not np.any(result == 3), 'A mean of probabilities must not invent a class absent in both label maps'
    assert uncertain[0, 0] and not uncertain[5, 6]
    assert all(b <= a+.5 for a, b in zip(qa['energy_history'], qa['energy_history'][1:]))


def test_HU_edge_discourages_boundary_crossing_without_enforcing_connected_shapes():
    p = np.zeros((8, 12, 14), np.float32); p[0] = .1; p[1] = .45; p[2] = .45
    labels = np.ones((12, 14), np.uint8); labels[:, 7:] = 2
    p[1, :, :7] = .85; p[2, :, :7] = .05
    p[2, :, 7:] = .85; p[1, :, 7:] = .05
    hu = np.zeros((12, 14), np.float32); hu[:, 7:] = 500
    result, _, _ = refine_plane(p, hu, labels, labels, PROTOCOL)
    np.testing.assert_array_equal(result, labels)


def test_slice_selection_is_central_and_does_not_count_border_components():
    a = np.zeros((30, 30, 20), np.uint8)
    a[2:10, 2:10, 9] = 1; a[12:20, 12:20, 9] = 2
    a[0:8, 0:8, 10] = 3
    chosen, rows = select_slice(a, a.copy(), [1, 1, 1], PROTOCOL)
    assert chosen['index'] == 9 and chosen['eligible_classes'] == 2
    assert all(8 <= r['index'] <= 12 for r in rows)
