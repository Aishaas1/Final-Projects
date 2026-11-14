"""Integrity regression tests: modified images must not silently pass a saved manifest."""
import hashlib
import pandas as pd
import pytest
from PIL import Image

from facejpeg.data import validate_manifest


def _fixture(tmp_path, with_hashes=True):
    rows = []
    for split in ("train", "val", "test"):
        for label, cls in ((0, "real"), (1, "fake")):
            path = tmp_path / "images" / split / cls / "face.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (10, 10), (label * 100, len(split) * 25, 30)).save(path)
            rel = path.relative_to(tmp_path).as_posix()
            row = {"path": rel, "label": label, "split": split,
                   "group": f"{split}-{cls}", "pixel_sha256": hashlib.sha256(rel.encode()).hexdigest()}
            if with_hashes:
                row["prepared_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "manifest.csv", index=False)
    return rows


def test_integrity_valid_images(tmp_path):
    _fixture(tmp_path)
    assert len(validate_manifest(tmp_path, verify_files=True, require_hashes=True)) == 6


def test_integrity_detects_changed_image(tmp_path):
    rows = _fixture(tmp_path)
    path = tmp_path / rows[0]["path"]
    Image.new("RGB", (10, 10), (255, 255, 255)).save(path)
    with pytest.raises(ValueError, match="Prepared image changed"):
        validate_manifest(tmp_path, verify_files=True)


def test_integrity_detects_invalid_hash(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "manifest.csv"
    text = path.read_text()
    path.write_text(text.replace("prepared_sha256", "prepared_sha256", 1).replace(
        pd.read_csv(path).prepared_sha256.iloc[0], "not-a-hash", 1))
    with pytest.raises(ValueError, match="Malformed"):
        validate_manifest(tmp_path)


def test_legacy_manifest_is_backward_compatible_but_not_certified(tmp_path):
    _fixture(tmp_path, with_hashes=False)
    assert len(validate_manifest(tmp_path, verify_files=True)) == 6
    with pytest.raises(ValueError, match="Legacy manifest"):
        validate_manifest(tmp_path, verify_files=True, require_hashes=True)
