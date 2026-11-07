import json
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest
import torch
from PIL import Image

from facejpeg.common import PREPROCESS, build_model, load_bundle, read_rgb, seed_everything, tensorize
from facejpeg.data import prepare
from facejpeg.features import cached_features, FeatureRows
from facejpeg.runtime import Monitor
from facejpeg.train import configure_transfer, train
from scripts.make_demo import make_demo


def test_cpu_cache_matches_backbone_reuses_and_excludes_test(tmp_path):
    raw = make_demo(tmp_path / "raw", 8)
    root = tmp_path / "prepared"
    frame = prepare(raw, root)
    seed_everything(42)
    model = build_model(False)
    configure_transfer(model, True, 0, False)
    out = tmp_path / "run"
    with Monitor(out) as monitor:
        cache, path = cached_features(model, frame, root, out, PREPROCESS, monitor, batch_size=4)
        train_frame = frame[frame.split == "train"].reset_index(drop=True)
        im = read_rgb(root / train_frame.path.iloc[0])
        with torch.no_grad():
            original = model.avgpool(model.features(tensorize(im).unsqueeze(0))).flatten(1)[0]
            mirrored = model.avgpool(model.features(tensorize(im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)).unsqueeze(0))).flatten(1)[0]
        assert torch.allclose(original, cache["train_original"][0], atol=1e-5, rtol=1e-5)
        assert torch.allclose(mirrored, cache["train_flipped"][0], atol=1e-5, rtol=1e-5)
        test_paths = set(frame.loc[frame.split == "test", "path"])
        assert not test_paths & set(cache["metadata"]["train_paths"] + cache["metadata"]["val_paths"])
        # Cached tensors must be ordinary tensors usable in classifier backward.
        x, y = FeatureRows(cache["train_original"], cache["train_labels"])[0]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model.classifier(x.unsqueeze(0)).flatten(), y.unsqueeze(0))
        loss.backward()
        forward = model.features.forward
        model.features.forward = Mock(side_effect=AssertionError("cache reuse must not rerun the backbone"))
        reused, reused_path = cached_features(model, frame, root, out, PREPROCESS, monitor)
        assert reused_path == path and torch.equal(reused["train_original"], cache["train_original"])
        model.features.forward = forward
        changed = root / train_frame.path.iloc[0]
        Image.new("RGB", (224, 224), "white").save(changed)
        modified, changed_path = cached_features(model, frame, root, out, PREPROCESS, monitor)
        assert changed_path != path
        assert modified["metadata"]["prepared_pixels_sha256"] != cache["metadata"]["prepared_pixels_sha256"]


def test_cached_training_resume_scheduler_and_frozen_backbone(tmp_path):
    raw = make_demo(tmp_path / "raw", 8)
    root = tmp_path / "prepared"
    prepare(raw, root)
    options = dict(quick=True, pretrained=False, batch_size=4, cache_features=True,
                   warmup_epochs=0, finetune_blocks=0, use_scheduler=True)
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    train(root, resumed, epochs=1, **options)
    train(root, resumed, epochs=2, **options)
    train(root, full, epochs=2, **options)
    a = torch.load(resumed / "last.pt", weights_only=True)
    b = torch.load(full / "last.pt", weights_only=True)
    assert a["scheduler_state"] == b["scheduler_state"]
    assert all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"])
    seed_everything(42)
    initial = build_model(False).state_dict()
    for key in a["model_state"]:
        if key.startswith("features."):
            assert torch.equal(initial[key], a["model_state"][key])
    assert any(not torch.equal(initial[k], a["model_state"][k]) for k in initial if k.startswith("classifier."))
    import pandas as pd
    predictions = pd.read_csv(resumed / "validation_predictions.csv")
    loaded, _ = load_bundle(resumed / "best.pt", "cpu")
    with torch.inference_mode():
        score = float(loaded(tensorize(read_rgb(root / predictions.path.iloc[0])).unsqueeze(0)).flatten().sigmoid()[0])
    assert abs(score - predictions.score_fake.iloc[0]) < 1e-5
    assert a["data_kind"] == "procedural_fixture" and a["run_label"] == "SMOKE_ONLY_NOT_A_FACE_DETECTOR"
    assert json.loads((resumed / "feature_cache_info.json").read_text())["test_images_cached"] is False


def test_cpu_notebook_configuration_runs_without_gpu(tmp_path, monkeypatch):
    nb = nbformat.read("FaceJPEG_CPU_Training.ipynb", as_version=4)
    nbformat.validate(nb)
    cells = [c.source for c in nb.cells if c.cell_type == "code"]
    assert len(cells) == 10 and 'MODE = "REAL_CPU"' in cells[2]
    assert 'USE_DRIVE = True' in cells[2]
    # Test config without Drive OAuth; does not claim hosted Colab or actual face training.
    source = cells[2].replace("USE_DRIVE = True", "USE_DRIVE = False")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    scope = {"torch": torch, "os": __import__("os"), "Path": Path, "PROJECT": tmp_path, "IN_COLAB": False}
    exec(compile(source, "cpu_config", "exec"), scope)
    assert scope["CPU_REAL"] and not scope["QUICK"]
    assert scope["MAX_PER_CLASS"] == 1000 and scope["EPOCHS"] == 20
    assert scope["BATCH_SIZE"] == 64
    assert 'cache_features=CPU_REAL' in cells[5]


def test_cache_requires_frozen_backbone(tmp_path):
    model = build_model(False)
    with Monitor(tmp_path) as monitor, pytest.raises(ValueError, match="frozen backbone"):
        cached_features(model, None, tmp_path, tmp_path, PREPROCESS, monitor)


def test_nested_project_fixture_cannot_be_relabelled_as_research_faces(tmp_path):
    raw = make_demo(tmp_path / "raw", 8)
    project = tmp_path / "project"
    prepare(raw, project / "artifacts/demo/prepared")
    output = tmp_path / "attempted-research"
    prepare(project, output, data_kind="research_faces")
    assert json.loads((output / "audit.json").read_text())["data_kind"] == "procedural_fixture"
    with pytest.raises(ValueError, match="Real Training"):
        train(output, tmp_path / "bad-training", cache_features=True,
              finetune_blocks=0, warmup_epochs=0, allow_cpu_real=True)
