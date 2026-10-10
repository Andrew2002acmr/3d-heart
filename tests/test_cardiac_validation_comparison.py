"""Protect validation-only, same-source comparisons from misleading metric deltas."""
from copy import deepcopy
import pytest
from heart3d.ml.cardiac_data import write_json
from scripts.compare_cardiac_validation import compare


def metrics(score):
    return {"partition": "validation", "grid": "original release NIfTI grid, probabilities restored before argmax",
            "cohort_SHA256": "cohort", "split_SHA256": "split", "checkpoint_SHA256": str(score),
            "records": [{"case_id": "a", "macro_foreground_Dice": score,
                         "classes": {name: {"target_voxels": 10, "Dice": score}
                                     for name in ("LV", "RV", "LA", "RA", "MYO", "AO", "PA")}}]}


def test_paired_delta_and_no_overwrite(tmp_path):
    baseline, variant, output = [tmp_path/name for name in ("baseline.json", "variant.json", "delta.json")]
    write_json(baseline, metrics(.7))
    write_json(variant, metrics(.8))
    result = compare(baseline, variant, output)
    assert result["mean_paired_delta"] == pytest.approx(.1)
    assert result["per_class"]["PA"]["mean_delta"] == pytest.approx(.1)
    assert not result["test_evaluated"] and result["cases_improved"] == 1
    with pytest.raises(ValueError, match="exists"):
        compare(baseline, variant, output)


@pytest.mark.parametrize("mutation,reason", [
    (lambda m: m.update(partition="test"), "validation"),
    (lambda m: m.update(grid="prepared grid384"), "original-grid"),
    (lambda m: m.update(split_SHA256="other"), "protocol"),
    (lambda m: m["records"][0].update(case_id="b"), "cases"),
    (lambda m: m["records"][0]["classes"]["PA"].update(target_voxels=20), "counts"),
])
def test_rejects_noncomparable_evaluations(tmp_path, mutation, reason):
    a, b, output = [tmp_path/name for name in ("a.json", "b.json", "output.json")]
    write_json(a, metrics(.7))
    changed = deepcopy(metrics(.8))
    mutation(changed)
    write_json(b, changed)
    with pytest.raises(ValueError, match=reason):
        compare(a, b, output)
    assert not output.exists()
