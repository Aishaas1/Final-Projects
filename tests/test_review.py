import ast
import json
from pathlib import Path

import pytest
import torch

from facejpeg.common import build_model, load_bundle
from facejpeg.data import inventory, prepare, safe_extract, validate_manifest
from facejpeg.evaluate import QUALITIES, binary_metrics, evaluate
from facejpeg.runtime import Monitor, loader_options
from facejpeg.train import configure_transfer, train
from scripts.download_data import classify_download_error, download
from scripts.make_demo import make_demo


def test_binary_metrics_known_values_and_no_positive_predictions():
    m = binary_metrics([0, 0, 1, 1], [.1, .8, .9, .4])
    for key in ["binary_accuracy_at_0_5_all", "binary_precision_at_0_5", "binary_recall_at_0_5", "binary_f1_at_0_5"]:
        assert m[key] == .5
    assert m["roc_auc_all"] == .75
    zero = binary_metrics([0, 1], [.1, .1])
    assert zero["binary_precision_at_0_5"] == zero["binary_f1_at_0_5"] == 0
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [.1, float("nan")])


@pytest.mark.parametrize("message,reason", [
    ("Tunnel connection failed: 403 Forbidden (proxy)", "network_policy"),
    ("401 Unauthorized", "authentication"), ("403 Forbidden", "access_denied"),
    ("404 Not found", "not_found"), ("SSL certificate verify failed", "tls")])
def test_download_error_diagnosis(message, reason):
    assert classify_download_error(RuntimeError(message))[0] == reason


def test_download_checks_class_inventory_and_does_not_log_sdk_secrets(tmp_path, monkeypatch):
    import kagglehub
    raw = make_demo(tmp_path / "raw", count=3)
    monkeypatch.setattr(kagglehub, "dataset_download", lambda _: str(raw))
    record = tmp_path / "download.json"
    assert download(record) == raw
    assert json.loads(record.read_text())["counts"]["real"] == 3

    def fail(_):
        raise RuntimeError("403 Forbidden proxy; ?credential=do-not-copy-this")

    monkeypatch.setattr(kagglehub, "dataset_download", fail)
    with pytest.raises(RuntimeError, match="وكيل الشبكة"):
        download(record)
    assert json.loads(record.read_text())["status"] == "failed"
    assert "do-not-copy-this" not in record.read_text()
    assert "do-not-copy-this" not in (tmp_path / "download_monitor/events.log").read_text()


def test_zip_and_drive_style_inventory_and_manifest_tampering(tmp_path):
    import zipfile
    raw = make_demo(tmp_path / "raw", 8)
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for p in raw.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(raw))
    restored = safe_extract(zpath, tmp_path / "restored")
    assert inventory(restored)[1] == inventory(raw)[1]
    out = tmp_path / "prepared"
    frame = prepare(restored, out)  # Marker prevents accidental research label.
    assert json.loads((out / "audit.json").read_text())["data_kind"] == "procedural_fixture"
    one = frame[frame.split == "train"].index[0]
    other = frame[frame.split == "test"].index[0]
    frame.loc[one, "group"] = frame.loc[other, "group"]
    frame.to_csv(out / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(out)


def test_monitor_records_failure_and_never_uses_a_training_timeout(tmp_path):
    with pytest.raises(ValueError):
        with Monitor(tmp_path, interval=.01) as monitor:
            monitor.stage("unit-stage", 10)
            monitor.advance(2, "two images processed")
            raise ValueError("controlled failure")
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == "failed" and state["completed"] == 2
    assert state["last_success"] == "two images processed" and state["eta_seconds"] is not None
    assert "controlled failure" not in (tmp_path / "events.log").read_text()


def test_literal_na_identity_is_preserved_for_grouping(tmp_path):
    import pandas as pd
    raw = make_demo(tmp_path / "raw", 8)
    md = tmp_path / "metadata.csv"
    pd.DataFrame([{"path": f"real/fixture_{i:03d}.png", "identity": "NA"} for i in [0, 1]]).to_csv(md, index=False)
    frame = prepare(raw, tmp_path / "prepared", metadata=md)
    selected = frame[frame.identity == "NA"]
    assert len(selected) == 2 and selected.group.nunique() == 1
    restored = validate_manifest(tmp_path / "prepared")
    assert int((restored.identity == "NA").sum()) == 2


def test_transfer_learning_and_loader_configuration():
    model = build_model(False)
    configure_transfer(model, quick=False, blocks=3, warmup=True)
    assert not any(p.requires_grad for p in model.features.parameters())
    configure_transfer(model, quick=False, blocks=3, warmup=False)
    assert all(p.requires_grad for p in model.features[-1].parameters())
    assert not any(p.requires_grad for p in model.features[0].parameters())
    assert loader_options("cpu", True) == {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
    assert loader_options("cuda", False, workers=2)["pin_memory"]


def test_new_training_resume_metrics_and_jpeg_pipeline(tmp_path):
    root, prepared = tmp_path / "raw", tmp_path / "prepared"
    make_demo(root, 8)
    prepare(root, prepared, data_kind="procedural_fixture")
    resumed, full = tmp_path / "resumed", tmp_path / "continuous"
    args = dict(quick=True, batch_size=4, pretrained=False)
    train(prepared, resumed, epochs=1, **args)
    train(prepared, resumed, epochs=2, **args)
    train(prepared, full, epochs=2, **args)
    a = torch.load(resumed / "last.pt", weights_only=True)
    b = torch.load(full / "last.pt", weights_only=True)
    for ra, rb in zip(a["history"], b["history"]):
        for key in ["train_loss", "val_loss", "train_accuracy_at_0_5", "val_accuracy_at_0_5"]:
            assert ra[key] == rb[key]
    assert all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"])
    _, bundle = load_bundle(resumed / "best.pt")
    assert bundle["run_label"] == "SMOKE_ONLY_NOT_A_FACE_DETECTOR"
    assert bundle["training_config"]["amp"] is False
    table = evaluate(prepared, resumed / "best.pt", tmp_path / "evaluation", device="cpu", batch_size=4)
    assert table.quality.tolist() == [str(q) for q in QUALITIES]
    for name in ["roc_curves.png", "jpeg_changes.csv", "REPORT_AR.md"]:
        assert (tmp_path / "evaluation" / name).is_file()
    with pytest.raises(ValueError, match="Real Training"):
        train(prepared, tmp_path / "invalid-real", quick=False)
    with pytest.raises(ValueError, match="Resume settings changed"):
        train(prepared, resumed, epochs=3, quick=True, batch_size=8, pretrained=False)


def test_final_notebook_syntax_and_preserves_colab_torch():
    import nbformat
    notebook = nbformat.read("FaceJPEG_Final_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    codes = [c.source for c in notebook.cells if c.cell_type == "code"]
    assert len(codes) == 10
    for code in codes:
        ast.parse(code)
    req = Path("requirements-colab.txt").read_text().splitlines()
    assert not any(line.startswith(("torch", "torchvision")) for line in req)
    assert "requirements-colab.txt" in codes[1]


def test_missing_weights_ui_is_clear_and_checkpoint_thresholds_validated(tmp_path):
    from app import create_app
    app = create_app(tmp_path / "missing.pt")
    assert "لا توجد أوزان" in str(app.config)
    app.close()
    checkpoint = tmp_path / "bad.pt"
    torch.save({"schema_version": 1, "architecture": "mobilenet_v3_small", "classes": ["real", "fake"],
                "thresholds": {"low": .9, "high": .1}}, checkpoint)
    with pytest.raises(ValueError, match="thresholds"):
        load_bundle(checkpoint)
