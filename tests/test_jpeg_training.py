"""JPEG training regression tests: leakage, cached recipes, resume, and test lock."""
import json
from unittest.mock import Mock

import nbformat
import pytest
import torch

from facejpeg.augmentation import jpeg_settings, jpeg_training_view
from facejpeg.common import PREPROCESS, build_model, file_sha256, read_rgb, seed_everything, tensorize
from facejpeg.data import prepare
from facejpeg.features import FeatureRows, cached_features
from facejpeg.runtime import Monitor
from facejpeg.train import Faces, configure_transfer, train
from scripts.make_demo import make_demo


@pytest.mark.parametrize("qualities,probability", [([0], .5), ([101], .5), ([True], .5),
    ([50.5], .5), ("90", .5), ([90], -.1), ([90], 1.1), ([90], float("nan"))])
def test_invalid_jpeg_training_settings_fail_before_training(qualities, probability):
    with pytest.raises(ValueError):
        jpeg_settings(qualities, probability)


def test_image_path_and_cached_recipe_use_independent_canonical_jpeg(tmp_path):
    raw = make_demo(tmp_path / "raw", 8)
    root = tmp_path / "prepared"
    frame = prepare(raw, root)
    train_frame = frame.query("split == 'train'").reset_index(drop=True)
    clean, _ = Faces(train_frame, root)[0]
    augmented, label = Faces(train_frame, root, jpeg_qualities=[30], jpeg_probability=1)[0]
    original = read_rgb(root / train_frame.path.iloc[0])
    expected = tensorize(jpeg_training_view(original, 30, PREPROCESS))
    assert torch.equal(augmented, expected) and not torch.equal(augmented, clean)
    assert label.item() == train_frame.label.iloc[0]
    unchanged, _ = Faces(train_frame, root, jpeg_qualities=[30], jpeg_probability=0)[0]
    assert torch.equal(clean, unchanged)
    assert jpeg_settings([90, 30, 90], .5) == ((90, 30), .5)


def test_cached_jpeg_is_train_only_matches_direct_model_and_reuses(tmp_path):
    root = tmp_path / "prepared"
    frame = prepare(make_demo(tmp_path / "raw", 8), root)
    before = {row.path: file_sha256(root / row.path) for row in frame.itertuples()}
    manifest_hash = file_sha256(root / "manifest.csv")
    seed_everything(42)
    model = build_model(False)
    configure_transfer(model, True, 0, False)
    with Monitor(tmp_path / "run") as monitor:
        base, base_path = cached_features(model, frame, root, tmp_path / "run", PREPROCESS, monitor, batch_size=4)
        cache, path = cached_features(model, frame, root, tmp_path / "run", PREPROCESS, monitor,
                                      batch_size=4, jpeg_qualities=[90, 30])
        assert path != base_path and cache["metadata"]["jpeg_qualities"] == [90, 30]
        assert torch.equal(base["val_original"], cache["val_original"])
        assert not any(key.startswith("val_jpeg") or key.startswith("test_") for key in cache)
        train_frame = frame.query("split == 'train'").reset_index(drop=True)
        im = read_rgb(root / train_frame.path.iloc[0])
        with torch.no_grad():
            for index, quality in enumerate([90, 30]):
                x = tensorize(jpeg_training_view(im, quality, PREPROCESS)).unsqueeze(0)
                expected = model.avgpool(model.features(x)).flatten(1)[0]
                flipped = model.avgpool(model.features(x.flip(-1))).flatten(1)[0]
                torch.testing.assert_close(cache["train_jpeg_original"][index, 0], expected, atol=1e-12, rtol=1e-4)
                torch.testing.assert_close(cache["train_jpeg_flipped"][index, 0], flipped, atol=1e-12, rtol=1e-4)
        forward = model.features.forward
        model.features.forward = Mock(side_effect=AssertionError("must reuse matching JPEG features"))
        reused, reused_path = cached_features(model, frame, root, tmp_path / "run", PREPROCESS, monitor,
                                              batch_size=4, jpeg_qualities=[90, 30])
        assert reused_path == path and reused["cache_reused"]
        model.features.forward = forward
    assert file_sha256(root / "manifest.csv") == manifest_hash
    assert before == {row.path: file_sha256(root / row.path) for row in frame.itertuples()}
    assert not cache["train_jpeg_original"].is_inference()


def test_feature_selection_preserves_image_identity_and_labels():
    original = torch.zeros(3, 2)
    jpeg = torch.stack([torch.arange(6).reshape(3, 2).float() + 10 * q for q in [1, 2]])
    labels = torch.tensor([0., 1., 0.])
    rows = FeatureRows(original, labels, jpeg_original=jpeg, jpeg_probability=1.)
    seed_everything(42)
    seen = set()
    for _ in range(40):
        x, y = rows[1]
        assert y.item() == 1 and any(torch.equal(x, jpeg[q, 1]) for q in range(2))
        seen.add(float(x[0]))
    assert len(seen) == 2
    unchanged = FeatureRows(original, labels, jpeg_original=jpeg, jpeg_probability=0.)
    assert torch.equal(unchanged[1][0], original[1])


def test_jpeg_cached_resume_is_exact_and_rejects_changed_probability(tmp_path):
    root = tmp_path / "prepared"
    prepare(make_demo(tmp_path / "raw", 8), root)
    options = dict(quick=True, pretrained=False, batch_size=4, cache_features=True, use_scheduler=True,
                   finetune_blocks=0, warmup_epochs=0, jpeg_qualities=[90, 30], jpeg_probability=.5)
    train(root, tmp_path / "resumed", epochs=1, **options)
    train(root, tmp_path / "resumed", epochs=2, **options)
    train(root, tmp_path / "full", epochs=2, **options)
    a = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    b = torch.load(tmp_path / "full/last.pt", weights_only=True)
    assert all(torch.equal(a["model_state"][key], b["model_state"][key]) for key in a["model_state"])
    assert a["scheduler_state"] == b["scheduler_state"]
    assert a["training_config"]["jpeg_qualities"] == [90, 30]
    previous = file_sha256(tmp_path / "resumed/best.pt")
    with pytest.raises(ValueError, match="Resume settings changed"):
        train(root, tmp_path / "resumed", epochs=2, **{**options, "jpeg_probability": .2})
    assert file_sha256(tmp_path / "resumed/best.pt") == previous


def test_pair_calibrates_both_before_test_and_rejects_tampering(tmp_path, monkeypatch):
    import facejpeg.experiment as experiment
    root, out = tmp_path / "prepared", tmp_path / "pair"
    frame = prepare(make_demo(tmp_path / "raw", 8), root)
    with pytest.raises(RuntimeError, match="لم يُفتح الاختبار"):
        experiment.evaluate_jpeg_comparison(root, out)
    actual_evaluate = experiment.evaluate
    blocked_evaluate = Mock(side_effect=AssertionError("training must never inspect test predictions"))
    monkeypatch.setattr(experiment, "evaluate", blocked_evaluate)
    options = dict(epochs=2, quick=True, pretrained=False, batch_size=4,
                   cache_features=True, finetune_blocks=0, warmup_epochs=0)
    checkpoints = experiment.fit_jpeg_comparison(root, out, jpeg_qualities=[90, 30], **options)
    blocked_evaluate.assert_not_called()
    lock = json.loads((out / "models_locked.json").read_text())
    assert lock["locked_before_test"] and not lock["test_images_used_for_optimization"]
    hashes = {name: file_sha256(path) for name, path in checkpoints.items()}
    repeat = experiment.fit_jpeg_comparison(root, out, jpeg_qualities=[90, 30], **options)
    assert hashes == {name: file_sha256(path) for name, path in repeat.items()}
    with pytest.raises(ValueError, match="Comparison plan changed"):
        experiment.fit_jpeg_comparison(root, out, jpeg_qualities=[50], **options)
    monkeypatch.setattr(experiment, "evaluate", actual_evaluate)
    table = experiment.evaluate_jpeg_comparison(root, out)
    assert len(table) == 7 and set(table.n_total) == {len(frame.query("split == 'test'"))}
    test_paths = set(frame.loc[frame.split == "test", "path"])
    import pandas as pd
    for name in checkpoints:
        protocol = json.loads((out / name / "evaluation/evaluation_protocol.json").read_text())
        assert protocol["thresholds"] == lock["models"][name]["thresholds"]
        predictions = pd.read_csv(out / name / "evaluation/predictions.csv")
        assert set(predictions.path) == test_paths
    assert json.loads((out / "comparison_protocol.json").read_text())["not_a_face_benchmark"]
    checkpoints["jpeg_augmented"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="locked checkpoint changed"):
        experiment.evaluate_jpeg_comparison(root, out)


def test_jpeg_notebook_is_self_contained_and_keeps_test_after_both_models():
    book = nbformat.read("FaceJPEG_JPEG_Training.ipynb", as_version=4)
    nbformat.validate(book)
    code = [cell.source for cell in book.cells if cell.cell_type == "code"]
    assert len(code) == 10
    for index, source in enumerate(code):
        compile(source, f"jpeg_notebook_stage_{index+1}", "exec")
    assert 'USE_DRIVE = False' in code[2] and 'MODE = "REAL_CPU"' in code[2]
    smoke = nbformat.read("FaceJPEG_JPEG_Smoke_Test.ipynb", as_version=4)
    smoke_code = [cell.source for cell in smoke.cells if cell.cell_type == "code"]
    assert 'MODE = "QUICK"' in smoke_code[2] and 'ALLOW_SOFTWARE_PREVIEW = True' in smoke_code[2]
    assert "fit_jpeg_comparison(" in code[5] and "evaluate(" not in code[5]
    assert "evaluate(" not in code[6] and "evaluate_jpeg_comparison(" in code[7]
    assert '"feature_cache" not in' in code[8]
