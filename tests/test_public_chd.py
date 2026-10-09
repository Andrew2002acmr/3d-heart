from pathlib import Path
import struct
from types import SimpleNamespace
import zlib
import pytest
from scripts.fetch_chd68_full import extract
from scripts.audit_public_chd import diagnoses

def test_split_entry_crosses_disk_boundary_and_existing_file_verified(tmp_path):
    payload=b"original CT bytes"*400;compressor=zlib.compressobj(wbits=-15)
    compressed=compressor.compress(payload)+compressor.flush();crc=zlib.crc32(payload)
    name=b"ct_1001_image.nii.gz"
    header=struct.pack("<4s5H3I2H",b"PK\x03\x04",20,0,8,0,0,crc,len(compressed),len(payload),len(name),0)
    joined=header+name+compressed;cut=len(header)+len(name)+5
    parts=[tmp_path/"p1",tmp_path/"p2"]
    parts[0].write_bytes(joined[:cut]);parts[1].write_bytes(joined[cut:])
    entry=SimpleNamespace(volume=0,header_offset=0,compress_size=len(compressed),file_size=len(payload),CRC=crc,compress_type=8)
    target=tmp_path/"out"
    report=extract(parts,entry,target)
    assert target.read_bytes()==payload and report["crc32"]==f"{crc:08x}"
    assert extract(parts,entry,target)["sha256"]==report["sha256"]
    target.write_bytes(b"foreign data")
    with pytest.raises(ValueError):extract(parts,entry,target)
    assert target.read_bytes()==b"foreign data"

def test_published_diagnoses_preserve_explicit_normal_and_unknown():
    rows=diagnoses("1001 | 1 | "+" | ".join(["0"]*14)+"\n1040\t"+"\t".join(["0"]*14+["1"])+"\n1002 "+" ".join(["0"]*15))
    assert rows["ct_1001"]==["ASD"]
    assert rows["ct_1040"]==["Normal"]
    assert rows["ct_1002"]==[]


def test_real_markdown_diagnoses_with_spaced_outer_pipes():
    assert diagnoses("| 1001 | 0 | 1 | " + " | ".join(["0"]*13) + " | ")["ct_1001"]==["AVSD"]


def metadata_workbook(path, rows):
    import zipfile
    from xml.sax.saxutils import escape
    ns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    body=[]
    for number, row in enumerate(rows, 1):
        cells=''.join(f'<c r="{col}{number}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>' for col, value in row.items())
        body.append(f'<row r="{number}">{cells}</row>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{ns}"><sheetData>{"".join(body)}</sheetData></worksheet>')


def metadata_row(index=1, birth="20201010", acquired="20221009", z="0.5"):
    return {"A":"idx", "B":index, "C":"PatientBirthDate", "D":birth,
            "E":"AcquisitionDate", "F":acquired, "J":"0.25", "K":"0.25", "M":z, "O":"synthetic scanner"}


def test_metadata_candidates_age_in_completed_years_without_raw_dates(tmp_path):
    import json
    from scripts.audit_public_chd_metadata import candidates
    path=tmp_path/"metadata.xlsx"
    metadata_workbook(path, [metadata_row()])
    item=candidates(path)["ct_1001"]
    assert item["age_candidate_completed_years"]==1
    assert item["age_candidate_days"]==729
    assert item["source_pixel_spacing_candidate_mm"]==[0.25, 0.25]
    assert not item["age_verified_for_release"] and not item["linkage_author_confirmed"]
    assert "20201010" not in json.dumps(item) and "20221009" not in json.dumps(item)


@pytest.mark.parametrize("rows, message", [
    ([{"A":"unexpected"}], "schema"),
    ([metadata_row(), metadata_row()], "Duplicate"),
])
def test_metadata_rejects_wrong_schema_and_ambiguous_index(tmp_path, rows, message):
    from scripts.audit_public_chd_metadata import candidates
    path=tmp_path/"metadata.xlsx";metadata_workbook(path, rows)
    with pytest.raises(ValueError, match=message):candidates(path)


@pytest.mark.parametrize("changes", [{"birth":"20231010"}, {"z":"nan"}, {"z":"0"}])
def test_metadata_invalid_age_or_spacing_remains_review(tmp_path, changes):
    from scripts.audit_public_chd_metadata import candidates
    path=tmp_path/"metadata.xlsx";metadata_workbook(path,[metadata_row(**changes)])
    item=candidates(path)["ct_1001"]
    assert item["metadata_values_status"]=="requires_review"
    assert "candidate_in_0_17" not in item
    assert not item["age_verified_for_release"]
