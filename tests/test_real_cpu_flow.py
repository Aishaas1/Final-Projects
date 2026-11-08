"""Notebook defaults and failure recovery must never reuse a toy face detector."""
from pathlib import Path
from unittest.mock import Mock
import os

import nbformat
import pytest
import torch


def codes():
    notebook = nbformat.read("FaceJPEG_Real_CPU_Ready.ipynb", as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def configured(tmp_path, source="zip", old_demo=None):
    scope = {"Path": Path, "os": os, "torch": torch, "PROJECT": tmp_path,
             "IN_COLAB": False, "demo": old_demo, "BEST": "stale-quick.pt", "TRAINING_READY": True}
    source_code = codes()[2].replace('SOURCE = "kaggle"', f'SOURCE = "{source}"', 1)
    exec(compile(source_code, "real_cpu_config", "exec"), scope)
    return scope


def test_real_cpu_default_clears_stale_weights_without_mounting_drive(tmp_path):
    previous_ui = Mock()
    scope = configured(tmp_path, old_demo=previous_ui)
    previous_ui.close.assert_called_once()
    assert scope["MODE"] == "REAL_CPU" and scope["CPU_REAL"] and not scope["QUICK"]
    assert scope["USE_DRIVE"] is False and scope["ALLOW_SOFTWARE_PREVIEW"] is False
    assert scope["BEST"] is None and scope["CHECKPOINTS"] == {}
    assert not scope["DATA_READY"] and not scope["TRAINING_READY"]
    assert scope["EPOCHS"] == 20 and scope["MAX_PER_CLASS"] == 1000


@pytest.mark.parametrize("source", ["zip", "kaggle"])
def test_missing_data_skips_training_metrics_export_and_ui(tmp_path, monkeypatch, capsys, source):
    import app
    import facejpeg.experiment as experiment
    import scripts.download_data as download_module
    import scripts.make_demo as demo_module
    scope = configured(tmp_path, source=source)
    # Rerunning data preparation must also invalidate a model from an earlier run.
    scope.update(TRAINING_READY=True, BEST="stale-quick.pt", CHECKPOINTS={"jpeg_augmented": "stale-quick.pt"})
    fitting = Mock(side_effect=AssertionError("No fitting without real data"))
    evaluation = Mock(side_effect=AssertionError("No test without a trained model"))
    create_ui = Mock(side_effect=AssertionError("No UI without genuine weights"))
    fixture = Mock(side_effect=AssertionError("Never fall back to geometric fixtures"))
    monkeypatch.setattr(experiment, "fit_jpeg_comparison", fitting)
    monkeypatch.setattr(experiment, "evaluate_jpeg_comparison", evaluation)
    monkeypatch.setattr(app, "create_app", create_ui)
    monkeypatch.setattr(demo_module, "make_demo", fixture)
    monkeypatch.setattr(download_module, "download", Mock(side_effect=RuntimeError("Kaggle network blocked 403")))
    cell_code = codes()
    for number in [4, 6, 7, 8, 9, 10]:
        exec(compile(cell_code[number-1], f"missing_data_stage_{number}", "exec"), scope)
    assert not scope["DATA_READY"] and not scope["TRAINING_READY"]
    assert scope["BEST"] is None and scope["demo"] is None
    fitting.assert_not_called(); evaluation.assert_not_called()
    create_ui.assert_not_called(); fixture.assert_not_called()
    assert "لم تجهز بيانات الوجوه" in capsys.readouterr().out
    assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.zip"))


@pytest.mark.parametrize("kind", ["toy", "corrupt"])
def test_interface_rejects_toy_or_corrupt_even_if_ready_flag_is_stale(tmp_path, monkeypatch, kind):
    import app
    checkpoint = Path("artifacts/jpeg_training/quick/jpeg_augmented/best.pt")
    if kind == "corrupt":
        checkpoint = tmp_path / "bad.pt"
        checkpoint.write_bytes(b"not a model")
    factory = Mock(side_effect=AssertionError("Invalid model must never reach the UI factory"))
    monkeypatch.setattr(app, "create_app", factory)
    scope = {"TRAINING_READY": True, "BEST": checkpoint, "ALLOW_SOFTWARE_PREVIEW": False,
             "LAUNCH_UI": True, "IN_COLAB": False, "demo": None}
    exec(compile(codes()[9], "reject_bad_ui_model", "exec"), scope)
    factory.assert_not_called()
    assert scope["demo"] is None


def test_verified_face_model_branch_launches_only_after_readiness(monkeypatch):
    # Pure wiring stub: no fabricated research weights, face training, or accuracy claim.
    import app
    import facejpeg.infer as inference
    fake_ui = Mock()
    factory = Mock(return_value=fake_ui)
    monkeypatch.setattr(inference, "Predictor", Mock(return_value=Mock(face_classification_enabled=True)))
    monkeypatch.setattr(app, "create_app", factory)
    scope = {"TRAINING_READY": True, "BEST": "unit-stub-only.pt", "ALLOW_SOFTWARE_PREVIEW": False,
             "LAUNCH_UI": True, "IN_COLAB": False, "demo": None}
    exec(compile(codes()[9], "verified_ui_wiring", "exec"), scope)
    factory.assert_called_once_with("unit-stub-only.pt")
    fake_ui.launch.assert_called_once_with(share=False, debug=False)
