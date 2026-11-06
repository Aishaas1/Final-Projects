import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from facejpeg.common import canonical, decisions, jpeg_copy, read_rgb
from facejpeg.data import prepare, safe_extract
from facejpeg.evaluate import summary_metrics
from facejpeg.train import select_thresholds
from scripts.make_demo import make_demo


def test_split_groups_duplicates_identity_and_corruption(tmp_path):
    raw = make_demo(tmp_path / "raw", 12)
    source = raw / "real/fixture_000.png"
    source_image = read_rgb(source)
    source_image.save(raw / "real/copy.png")
    jpeg_copy(source_image, 90).save(raw / "real/compressed.png")
    (raw / "fake/broken.jpg").write_bytes(b"not an image")
    pd.DataFrame([
        {"path": "real/fixture_001.png", "identity": "person_1", "source_id": ""},
        {"path": "real/fixture_002.png", "identity": "person_1", "source_id": ""},
    ]).to_csv(tmp_path / "metadata.csv", index=False)
    out = tmp_path / "prepared"
    df = prepare(raw, out, tmp_path / "metadata.csv")
    assert df.groupby("group").split.nunique().max() == 1
    assert all(set(part.label) == {0, 1} for _, part in df.groupby("split"))
    identities = df[df.identity == "person_1"]
    assert identities.split.nunique() == 1 and len(identities) == 2
    a = df[df.source_path.isin(["real/fixture_000.png", "real/copy.png", "real/compressed.png"])]
    assert a.group.nunique() == 1 and len(a) == 2
    report = json.loads((out / "audit.json").read_text())
    assert report["exact_duplicates_removed"] == 1 and report["rejected"] == 1
    assert all(Image.open(out / path).size == (224, 224) for path in df.path)


def test_jpeg_independent_and_shared_preprocess():
    rng = np.random.default_rng(3)
    original = canonical(Image.fromarray(rng.integers(0, 255, (250, 250, 3), dtype=np.uint8)))
    before = np.asarray(original).copy()
    q30 = jpeg_copy(original, 30)
    jpeg_copy(original, 90)
    assert np.array_equal(np.asarray(q30), np.asarray(jpeg_copy(original, 30)))
    assert np.array_equal(before, np.asarray(original))
    assert not np.array_equal(before, np.asarray(q30))


def test_abstention_denominators_and_threshold_boundary():
    s = [.2, .8, .5, .2, .8, .5]
    y = [0, 0, 0, 1, 1, 1]
    d = decisions(s, {"low": .2, "high": .8})
    assert d.tolist() == [0, 1, 2, 0, 1, 2]
    m = summary_metrics(y, s, d)
    assert m["fake_detection_rate_all_fake"] == 1/3
    assert m["false_accusation_rate_all_real"] == 1/3
    assert m["abstain_total"] == 2 and m["coverage_all"] == 4/6
    assert m["selective_accuracy_accepted"] == .5
    thresholds, _ = select_thresholds(np.array([0, 0, 1, 1]), np.array([.1, .3, .7, .9]))
    assert 0 < thresholds["low"] < thresholds["high"] < 1


def test_zip_path_traversal_is_rejected(tmp_path):
    import zipfile
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escaped", "bad")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "output")
    assert not (tmp_path / "escaped").exists()


def test_arabic_errors_and_multiple_faces(tmp_path, monkeypatch):
    import facejpeg.infer as infer
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"broken")
    with pytest.raises(ValueError, match="تالف"):
        infer.validated_image(bad)
    valid = tmp_path / "valid.png"
    Image.new("RGB", (224, 224), "white").save(valid)
    with pytest.raises(ValueError, match="لم أجد"):
        infer.validated_image(valid)
    monkeypatch.setattr(infer, "face_count", lambda _: 2)
    with pytest.raises(ValueError, match="عدة وجوه"):
        infer.validated_image(valid)


def test_checkpoint_reload_and_ui_same_score(monkeypatch):
    checkpoint = Path("artifacts/smoke/best.pt")
    if not checkpoint.exists():
        pytest.skip("Run the documented smoke training to produce best.pt")
    import facejpeg.infer as infer
    from facejpeg.common import load_bundle, tensorize
    import torch
    path = next(Path("artifacts/smoke/prepared/images/test/real").glob("*.png"))
    im = read_rgb(path)
    model, bundle = load_bundle(checkpoint)
    with torch.inference_mode():
        direct = float(model(tensorize(im, bundle["preprocess"]).unsqueeze(0)).flatten().sigmoid()[0])
    predictor = infer.Predictor(checkpoint, "cpu")
    assert abs(predictor.score(im) - direct) < 1e-7
    monkeypatch.setattr(infer, "face_count", lambda _: 1)
    result = predictor.compare(path, 70)
    assert f"{direct:.4f}" in result[2]
    assert "اختبار برمجي" in result[4]
