"""Public TCIA access and bounded Heart-only RTSTRUCT extraction.

The bounded stream stops AFTER the complete Heart item. The resulting DICOM
is explicitly a derived ROI extract, never represented as the original full RT.
"""
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import pydicom
from pydicom.filereader import read_partial, read_dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid
from heart3d.storage import require_space

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"


def mark_derived_extract(dataset, receipt):
    """Give modified census objects new UIDs; source DICOM is never changed."""
    if 'source_rt_sop_uid' in receipt:
        return
    source_sop, source_series = str(dataset.SOPInstanceUID), str(dataset.SeriesInstanceUID)
    receipt.update(source_rt_sop_uid=source_sop, source_rt_series_uid=source_series)
    dataset.SOPInstanceUID = generate_uid(entropy_srcs=[source_sop, 'Heart-only census extract v1'])
    dataset.SeriesInstanceUID = generate_uid(entropy_srcs=[source_series, 'Heart-only census extract v1'])
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.SeriesDescription = 'Derived Heart-only census extract; not original RTSTRUCT'
    receipt.update(derived_sop_uid=str(dataset.SOPInstanceUID), derived_series_uid=str(dataset.SeriesInstanceUID))


def json_get(endpoint, **params):
    from urllib.parse import urlencode
    with urllib.request.urlopen(BASE + endpoint + "?" + urlencode(params), timeout=60) as r:
        return json.load(r)


def download(url, path, reserve_bytes=0):
    path = Path(path)
    if reserve_bytes:
        require_space(path.parent, 0, reserve_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(url, timeout=90) as r, partial.open("wb") as f:
            while block := r.read(1 << 20):
                if reserve_bytes:
                    require_space(path.parent, len(block), reserve_bytes)
                f.write(block)
        partial.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return {"url": url, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


class BoundedStream:
    """Seekable forward cache for pydicom over HTTP; bounded memory/network."""
    def __init__(self, source, limit=64 << 20, prefetch=0):
        self.source, self.limit, self.prefetch = source, limit, prefetch
        self.buffer, self.position = bytearray(), 0

    def tell(self):
        return self.position

    def _fill(self, end):
        if end > self.limit:
            raise ValueError("RTSTRUCT prefix budget exceeded; requires full RT review")
        while len(self.buffer) < end:
            block = self.source.read(min(max(end - len(self.buffer), self.prefetch),
                                         1 << 20, self.limit-len(self.buffer)))
            if not block:
                break
            self.buffer.extend(block)

    def read(self, size=-1):
        if size < 0:
            raise ValueError("Unbounded stream read is forbidden")
        self._fill(self.position + size)
        data = bytes(self.buffer[self.position:self.position + size])
        self.position += len(data)
        return data

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else None
        if target is None or target < 0:
            raise ValueError("Unsupported stream seek")
        self._fill(target)
        self.position = target
        return target


def extract_heart_stream(stream):
    """Parse header and complete Heart ROI using DICOM item delimiters."""
    header = read_partial(stream, stop_when=lambda tag, vr, length: tag == (0x3006, 0x0039))
    if not hasattr(header, "StructureSetROISequence"):
        raise ValueError("RTSTRUCT ROI definitions missing")
    matches = [r for r in header.StructureSetROISequence if str(r.ROIName).strip().casefold() == "heart"]
    if len(matches) > 1:
        raise ValueError("Ambiguous Heart ROI")
    roi = None
    if matches:
        syntax = header.file_meta.TransferSyntaxUID
        if syntax.is_compressed or syntax.is_deflated or not syntax.is_little_endian:
            raise ValueError("Unsupported RTSTRUCT stream transfer syntax")
        implicit = syntax.is_implicit_VR
        raw = stream.read(8 if implicit else 12)
        if len(raw) != (8 if implicit else 12) or struct.unpack_from("<HH", raw) != (0x3006, 0x0039):
            raise ValueError("ROIContourSequence not found at expected stop")
        if not implicit and raw[4:6] != b"SQ":
            raise ValueError("Expected explicit SQ")
        length = struct.unpack_from("<I", raw, 4 if implicit else 8)[0]
        end = None if length == 0xFFFFFFFF else stream.tell() + length
        while end is None or stream.tell() < end:
            item_header = stream.read(8)
            if len(item_header) != 8:
                raise ValueError("Truncated ROI sequence")
            group, element, item_length = struct.unpack("<HHI", item_header)
            if (group, element) == (0xFFFE, 0xE0DD):
                break
            if (group, element) != (0xFFFE, 0xE000):
                raise ValueError("Invalid DICOM sequence item")
            item = read_dataset(stream, implicit, True, bytelength=None if item_length == 0xFFFFFFFF else item_length,
                                at_top_level=False)
            if item_length == 0xFFFFFFFF:
                # read_dataset can return on premature EOF: require the actual delimiter.
                if bytes(stream.buffer[stream.tell()-8:stream.tell()]) != struct.pack('<HHI', 0xFFFE, 0xE00D, 0):
                    raise ValueError('Truncated undefined-length ROI item')
            if int(item.ReferencedROINumber) == int(matches[0].ROINumber):
                roi = item
                break
        if roi is None:
            raise ValueError("Heart definition exists but complete contour item missing")
    extracted = header
    extracted.ROIContourSequence = Sequence([roi] if roi is not None else [])
    # Preserve provenance separately; keep source SOP UID for reference validation.
    return extracted, {"ROI_names": [str(r.ROIName) for r in header.StructureSetROISequence],
        "heart_defined": bool(matches), "heart_contour_item_complete": roi is not None,
        "heart_contours": len(getattr(roi, "ContourSequence", [])),
        "derived_roi_extract": True, "original_full_RTSTRUCT_downloaded": False}


def fetch_heart_extract(series_uid, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target, receipt_path = destination / "heart_extract.dcm", destination / "heart_extract_receipt.json"
    if target.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest() != receipt["derived_sha256"]:
            raise ValueError("Cached Heart extract differs from receipt")
        dataset = pydicom.dcmread(target)
        if 'source_rt_sop_uid' not in receipt:
            mark_derived_extract(dataset, receipt)
            dataset.save_as(target, enforce_file_format=True)
            receipt['derived_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
            receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        return dataset, receipt
    sops = json_get("getSOPInstanceUIDs", SeriesInstanceUID=series_uid)
    if len(sops) != 1:
        raise ValueError("Expected exactly one RTSTRUCT instance")
    url = BASE + f"getSingleImage?SeriesInstanceUID={series_uid}&SOPInstanceUID={sops[0]['SOPInstanceUID']}"
    with urllib.request.urlopen(url, timeout=90) as response:
        stream = BoundedStream(response, prefetch=64 << 10)
        dataset, receipt = extract_heart_stream(stream)
        receipt.update(url=url, received_prefix_bytes=len(stream.buffer), prefix_sha256=hashlib.sha256(stream.buffer).hexdigest())
    if str(dataset.SeriesInstanceUID) != series_uid:
        raise ValueError("RT series UID mismatch")
    mark_derived_extract(dataset, receipt)
    dataset.save_as(target, enforce_file_format=True)
    receipt["derived_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return dataset, receipt


def fetch_ct_probe(series_uid, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    sop_path = destination / "ct_sops.json"
    if not sop_path.exists():
        sop_path.write_text(json.dumps(json_get("getSOPInstanceUIDs", SeriesInstanceUID=series_uid)))
    sops = json.loads(sop_path.read_text())
    if not sops:
        raise ValueError("Empty CT series inventory")
    path = destination / "ct_probe.dcm"
    receipt = download(BASE + f"getSingleImage?SeriesInstanceUID={series_uid}&SOPInstanceUID={sops[0]['SOPInstanceUID']}", path)
    return pydicom.dcmread(path, stop_before_pixels=True), {r["SOPInstanceUID"] for r in sops}, receipt


def fetch_full_series(series_uid, destination, reserve_bytes=0):
    destination = Path(destination)
    archive = destination.parent / (destination.name + ".zip")
    receipt = download(BASE + "getImage?SeriesInstanceUID=" + series_uid, archive, reserve_bytes=reserve_bytes)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe ZIP path")
            if item.is_dir():
                continue
            payload = z.read(item)  # verifies original ZIP CRC
            if target.exists() and target.read_bytes() != payload:
                raise ValueError("Refuse to overwrite existing DICOM")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if reserve_bytes:
                    require_space(destination, len(payload), reserve_bytes)
                target.write_bytes(payload)
    receipt["original_zip_crc_verified"] = True
    return receipt
