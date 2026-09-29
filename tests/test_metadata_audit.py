from heart3d.metadata_audit import compare_keys, first_sheet_cells
import zipfile


def test_offset_is_evidence_not_authorized_patient_mapping():
    result = compare_keys([1001, 1003, 1010], [10, 1, 3], [1003, 1010, 1001])
    assert result["constant_offset_matching_all_keys"] == 1000
    assert result["classification_matches_file_ids"]
    assert not result["patient_row_correspondence_confirmed"]
    assert not result["apply_metadata_spacing"]


def test_mismatch_and_duplicates_visible():
    result = compare_keys([1001, 1002], [1, 1, 5], [1001])
    assert result["constant_offset_matching_all_keys"] is None
    assert result["duplicates"]["metadata"] == [1]
    assert not result["classification_matches_file_ids"]


def test_read_xlsx_values_and_actual_sheet_relationship(tmp_path):
    path = tmp_path / "fixture.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Geometry" r:id="rId3"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels", '<Relationships><Relationship Id="rId3" Target="worksheets/sheet3.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet3.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="4"><c r="B4"><v>10</v></c><c r="I4" t="inlineStr"><is><t>PixelSpacing</t></is></c></row></sheetData></worksheet>')
    sheet, rows = first_sheet_cells(path)
    assert sheet == "Geometry"
    assert rows == [(4, {"B": 10., "I": "PixelSpacing"})]
