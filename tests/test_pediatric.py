import pytest
from heart3d.pediatric import age_fields, clinical_status, dicom_age, read_hvsmr, validate_registry


def test_no_disease_flag_does_not_imply_normal():
    assert clinical_status(False, []) == "unknown"
    assert clinical_status(True, []) == "no_structural_cardiac_abnormality_reported"
    assert clinical_status(True, ["Glenn"]) == "pathological"


def test_age_rule_and_fractional_group_boundaries():
    assert age_fields(None)["eligible_by_reported_age"] is None
    assert age_fields(1.99)["eligible_by_reported_age"] is False
    assert age_fields(2)["age_group"] == "2-5"
    assert age_fields(5.9)["age_group"] == "2-5"
    assert age_fields(6)["age_group"] == "6-11"
    assert age_fields(11.9)["age_group"] == "6-11"
    assert age_fields(12)["age_group"] == "12-17"
    assert age_fields(17)["eligible_by_reported_age"] is True
    assert age_fields(17.01)["eligible_by_reported_age"] is False
    assert age_fields(18)["eligible_by_reported_age"] is False
    with pytest.raises(ValueError):
        age_fields(float("nan"))


def test_dicom_age_preserves_precision():
    assert dicom_age("024M")["age"] == 2
    assert dicom_age("005D")["eligible_by_reported_age"] is False
    assert dicom_age("017Y")["exact_age_available"] is False
    with pytest.raises(ValueError):
        dicom_age("17")


def test_hvsmr_missing_clinical_record_is_retained_and_not_eligible(tmp_path):
    clinical = tmp_path / "clinical.csv"
    clinical.write_text("Pat,Age,Category,Normal,Glenn\n0,17,severe,,X\n1,18,mild,X,\n", encoding="utf-8")
    technical = tmp_path / "technical.csv"
    technical.write_text("Pat,TE\n0,2.1\n1,1.7\n59,1.6\n,,,,\n", encoding="utf-8")
    rows = read_hvsmr(clinical, technical)
    validate_registry(rows)
    by = {r["patient_id"]: r for r in rows}
    assert by["pat0"]["previous_surgery"] == ["Glenn"]
    assert by["pat0"]["healthy_status"] == "pathological"
    assert by["pat1"]["eligible_by_reported_age"] is False
    assert by["pat59"]["age"] is None
    assert by["pat59"]["healthy_status"] == "unknown"
    assert by["pat59"]["eligible_by_reported_age"] is None
    assert by["pat59"]["binary_classification_eligible"] is False


def test_registry_rejects_duplicate_and_unconfirmed_classification():
    r = {"dataset": "test", "patient_id": "a", "age": None,
         "eligible_by_reported_age": None, "healthy_status": "unknown", "binary_classification_eligible": False}
    with pytest.raises(ValueError, match="Duplicate"):
        validate_registry([r, r])
    with pytest.raises(ValueError, match="Unconfirmed"):
        validate_registry([{**r, "binary_classification_eligible": True}])
