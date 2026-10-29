import copy
import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from heart3d.anatomy_contours import rasterize, validate_recipe
from heart3d.storage import sha256_file


def setup_source(tmp_path):
    affine = np.diag([-.7, -1.1, 1.8, 1.])
    affine[:3, 3] = [20., -30., 40.]
    source = nib.Nifti1Image(np.zeros((18, 24, 7), dtype=np.int16), affine)
    source.header.set_xyzt_units('mm')
    source.set_qform(affine, 1)
    source.set_sform(affine, 2)
    path = tmp_path / 'ct.nii.gz'
    nib.save(source, path)
    admission = {key: True for key in ('user_attested_permission_and_basis', 'local_only',
        'geometry_verified', 'independent_reader_verified', 'metadata_deidentified',
        'pixel_identifier_screening_passed')}
    admission['ct_sha256'] = sha256_file(path)
    admission_path = tmp_path / 'admission.json'
    admission_path.write_text(json.dumps(admission))
    recipe = {'schema_version': 1, 'ct_sha256': sha256_file(path), 'target': {'axis': 2, 'index': 3},
        'source': 'cloud_model', 'annotation_protocol': {'purpose': 'synthetic research test'},
        'labels': [{'label': 1, 'outer': [[2, 4], [6, 4], [6, 9], [2, 9]], 'holes': [],
                    'confidence': 'low', 'boundary_uncertainty': ['synthetic AV boundary']}]}
    return path, admission_path, recipe


def write_recipe(tmp_path, recipe):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    return path


def test_native_xy_centres_original_geometry_partial_scope_and_hashes(tmp_path):
    ct, admission, recipe = setup_source(tmp_path)
    recipe_path = write_recipe(tmp_path, recipe)
    original = nib.load(ct)
    digest = sha256_file(ct)
    result = rasterize(ct, admission, recipe_path, tmp_path / 'candidate')
    mask = nib.load(tmp_path / 'candidate/candidate.nii.gz')
    data = np.asanyarray(mask.dataobj)
    assert data.dtype == np.uint8 and data.shape == original.shape
    assert np.count_nonzero(data) == 5 * 6
    assert np.all(data[2:7, 4:10, 3] == 1) and data[7, 4, 3] == 0
    assert not data[:, :, :3].any() and not data[:, :, 4:].any()
    np.testing.assert_array_equal(mask.affine, original.affine)
    np.testing.assert_array_equal(mask.get_qform(), original.get_qform())
    np.testing.assert_array_equal(mask.get_sform(), original.get_sform())
    assert int(mask.header['qform_code']) == 1 and int(mask.header['sform_code']) == 2
    assert mask.header.get_xyzt_units()[0] == 'mm' and mask.dataobj.slope == 1 and mask.dataobj.inter == 0
    assert not mask.header.extensions
    assert all(not bytes(mask.header[name]).strip(b'\0 ') for name in ('data_type', 'db_name', 'descrip', 'aux_file', 'intent_name'))
    assert sha256_file(ct) == digest
    assert result['source'] == 'cloud_model' and not result['expert_approved']
    assert not result['medical_correctness_verified'] and not result['rasterizer_inference_performed']
    assert result['scope'] == {'axis': 2, 'index': 3, 'complete_volume': False}
    assert result['candidate']['sha256'] == sha256_file(tmp_path / 'candidate/candidate.nii.gz')
    assert (tmp_path / 'candidate/recipe.json').read_bytes() == recipe_path.read_bytes()
    expected = result.pop('manifest_sha256')
    canonical = json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected


def test_matching_hole_gives_lv_endocardial_boundary_and_deterministic_candidate(tmp_path):
    ct, admission, recipe = setup_source(tmp_path)
    recipe['labels'].append({'label': 5, 'outer': [[1, 3], [7, 3], [7, 10], [1, 10]],
        'holes': [copy.deepcopy(recipe['labels'][0]['outer'])], 'confidence': 'medium',
        'boundary_uncertainty': []})
    path = write_recipe(tmp_path, recipe)
    first = rasterize(ct, admission, path, tmp_path / 'first')
    second = rasterize(ct, admission, path, tmp_path / 'second')
    assert first['candidate']['sha256'] == second['candidate']['sha256']
    array = np.asanyarray(nib.load(tmp_path / 'first/candidate.nii.gz').dataobj)
    assert array[2, 4, 3] == 1 and array[1, 3, 3] == 5
    assert first['boundary_conflicts'] == []
    assert first['labels']['1']['saved_voxels'] == 30 and first['labels']['5']['saved_voxels'] == 26


def test_shared_boundary_explicit_ownership_is_order_independent(tmp_path):
    ct, _, recipe = setup_source(tmp_path)
    recipe['labels'].append({'label': 2, 'outer': [[6, 4], [10, 4], [10, 9], [6, 9]],
        'holes': [], 'confidence': 'low', 'boundary_uncertainty': []})
    plane, details, conflicts = validate_recipe(recipe, sha256_file(ct), (18, 24, 7))
    assert np.all(plane[6, 4:10] == 1) and np.all(plane[7:11, 4:10] == 2)
    assert conflicts == [{'labels': [1, 2], 'shared_boundary_voxels': 6, 'owner': 1}]
    recipe['labels'].reverse()
    reverse, _, reversed_conflicts = validate_recipe(recipe, sha256_file(ct), (18, 24, 7))
    np.testing.assert_array_equal(plane, reverse)
    assert reversed_conflicts == conflicts and details[2]['saved_voxels'] == 24


@pytest.mark.parametrize('case', ['duplicate_label', 'out_of_bounds', 'nonfinite', 'interior_overlap',
    'hole_outside', 'self_cross', 'wrong_hash', 'extra_field', 'bad_target', 'bool_label', 'empty_mask'])
def test_invalid_recipes_fail_before_output_creation(tmp_path, case):
    ct, admission, recipe = setup_source(tmp_path)
    row = recipe['labels'][0]
    if case == 'duplicate_label': recipe['labels'].append(copy.deepcopy(row))
    elif case == 'out_of_bounds': row['outer'][0][0] = -0.1
    elif case == 'nonfinite': row['outer'][0][0] = float('nan')
    elif case == 'interior_overlap':
        other = copy.deepcopy(row); other['label'] = 2; recipe['labels'].append(other)
    elif case == 'hole_outside': row['holes'] = [[[0, 0], [5, 0], [5, 5], [0, 5]]]
    elif case == 'self_cross': row['outer'] = [[2, 4], [7, 10], [2, 10], [6, 4]]
    elif case == 'wrong_hash': recipe['ct_sha256'] = '0' * 64
    elif case == 'extra_field': recipe['url'] = 'https://unused.invalid'
    elif case == 'bad_target': recipe['target']['axis'] = 1
    elif case == 'bool_label': row['label'] = True
    elif case == 'empty_mask': row['holes'] = [copy.deepcopy(row['outer'])]
    with pytest.raises(ValueError):
        rasterize(ct, admission, write_recipe(tmp_path, recipe), tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()


def test_missing_admission_and_existing_output_preserve_everything(tmp_path):
    ct, admission, recipe = setup_source(tmp_path)
    recipe_path = write_recipe(tmp_path, recipe)
    record = json.loads(admission.read_text()); record['geometry_verified'] = False
    admission.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='admitted'):
        rasterize(ct, admission, recipe_path, tmp_path / 'blocked')
    assert not (tmp_path / 'blocked').exists()
    out = tmp_path / 'existing'; out.mkdir(); marker = out / 'marker.txt'; marker.write_text('preserve')
    with pytest.raises(ValueError, match='existing'):
        rasterize(ct, admission, recipe_path, out)
    assert marker.read_text() == 'preserve'


def test_fractional_polygon_uses_voxel_centres_without_rounding_vertices(tmp_path):
    ct, _, recipe = setup_source(tmp_path)
    recipe['labels'][0]['outer'] = [[2.2, 4.2], [6.8, 4.2], [6.8, 9.8], [2.2, 9.8]]
    plane, _, _ = validate_recipe(recipe, sha256_file(ct), (18, 24, 7))
    assert np.count_nonzero(plane) == 4 * 5
    assert plane[2, 5] == 0 and plane[3, 5] == 1 and plane[6, 9] == 1 and plane[7, 9] == 0
