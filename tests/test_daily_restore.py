import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import nbformat
import pytest

from facejpeg.common import canonical, read_rgb
from facejpeg.daily import backup_checkpoint, discover_checkpoints, import_checkpoint, verify_checkpoint
from facejpeg.infer import Predictor

ROOT = Path(__file__).resolve().parents[1]
SOFTWARE_MODEL = ROOT / "artifacts/review/quick/best.pt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_backup_restore_retains_exact_model_bytes_metadata_and_raw_score(tmp_path):
    # Procedural fixture for software verification only; never represented as a face detector.
    backup = backup_checkpoint(SOFTWARE_MODEL, tmp_path / "saved.zip", require_faces=False)
    restored = import_checkpoint(backup, tmp_path / "restored/best.pt", require_faces=False)
    assert sha(restored) == sha(SOFTWARE_MODEL)
    p1, p2 = Predictor(SOFTWARE_MODEL, "cpu"), Predictor(restored, "cpu")
    image = canonical(read_rgb(ROOT / "examples/test_images/real_nasa.png"))
    assert p1.score(image) == p2.score(image)
    for key in ["preprocess", "thresholds", "classes", "data_kind"]:
        assert p1.bundle[key] == p2.bundle[key]
    assert not p2.face_classification_enabled
    report = json.loads((restored.parent / "loaded_model.json").read_text())
    assert report["checkpoint_sha256"] == sha(SOFTWARE_MODEL)
    with zipfile.ZipFile(backup) as archive:
        assert set(archive.namelist()) == {"best.pt", "backup_manifest.json", "README_AR.txt"}


def test_public_daily_rejects_software_weights_and_preserves_old_model(tmp_path):
    target = tmp_path / "models/best.pt"
    target.parent.mkdir()
    target.write_bytes(b"existing model must not be replaced")
    before = sha(target)
    with pytest.raises(ValueError, match="QUICK"):
        import_checkpoint(SOFTWARE_MODEL, target)
    assert sha(target) == before
    with pytest.raises(ValueError, match="QUICK"):
        backup_checkpoint(SOFTWARE_MODEL, tmp_path / "no.zip")
    assert not (tmp_path / "no.zip").exists()


@pytest.mark.parametrize("preference", ["baseline", "jpeg_augmented"])
def test_pair_zip_selects_explicit_preference_only_and_ignores_images(tmp_path, preference):
    archive_path = tmp_path / "results.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for model in ["baseline", "jpeg_augmented"]:
            archive.write(SOFTWARE_MODEL, f"my_run/{model}/best.pt")
        archive.writestr("raw/private-training-image.jpg", b"never extract this")
    restored = import_checkpoint(archive_path, tmp_path / "model/best.pt", preference, require_faces=False)
    report = json.loads((restored.parent / "loaded_model.json").read_text())
    assert report["source_member"] == f"my_run/{preference}/best.pt"
    assert sorted(p.name for p in restored.parent.iterdir()) == ["best.pt", "loaded_model.json"]


def test_root_selected_model_wins_over_old_pair_candidates(tmp_path):
    archive_path = tmp_path / "upgrade.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(SOFTWARE_MODEL, "best.pt")
        archive.write(SOFTWARE_MODEL, "candidates/jpeg_augmented/best.pt")
    restored = import_checkpoint(archive_path, tmp_path / "model/best.pt", require_faces=False)
    assert json.loads((restored.parent / "loaded_model.json").read_text())["source_member"] == "best.pt"


@pytest.mark.parametrize("names,match", [
    (["../best.pt"], "غير آمن"), (["/best.pt"], "غير آمن"),
    (["a/../../best.pt"], "غير آمن"),
    (["a/jpeg_augmented/best.pt", "b/jpeg_augmented/best.pt"], "أكثر"),
    (["x/best.pt", "y/z/best.pt"], "تحديد"), (["photos/face.png"], "لم أجد"),
])
def test_invalid_zip_member_paths_missing_or_ambiguous_model(tmp_path, names, match):
    archive_path = tmp_path / "invalid.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in names:
            archive.writestr(name, b"fake weights")
    with pytest.raises(ValueError, match=match):
        import_checkpoint(archive_path, tmp_path / "model/best.pt")
    assert not (tmp_path / "model/best.pt").exists()
    assert not (tmp_path / "best.pt").exists()


def test_tampered_backup_is_rejected(tmp_path):
    archive_path = tmp_path / "changed.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(SOFTWARE_MODEL, "best.pt")
        archive.writestr("backup_manifest.json", json.dumps({"checkpoint_sha256": "not the checksum"}))
    with pytest.raises(ValueError, match="بصمة"):
        import_checkpoint(archive_path, tmp_path / "model/best.pt", require_faces=False)


def test_invalid_manifest_is_reported_without_python_attribute_error(tmp_path):
    archive_path = tmp_path / "invalid_manifest.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(SOFTWARE_MODEL, "best.pt")
        archive.writestr("backup_manifest.json", "[]")
    with pytest.raises(ValueError, match="بيانات"):
        import_checkpoint(archive_path, tmp_path / "model/best.pt", require_faces=False)


@pytest.mark.parametrize("filename,content,match", [
    ("broken.zip", b"incomplete", "ZIP"), ("bad.pt", b"not a pytorch model", "تالف"),
    ("photo.png", b"not weights", "الصورة|صورة"),
])
def test_invalid_upload_reports_actionable_arabic(tmp_path, filename, content, match):
    source = tmp_path / filename
    source.write_bytes(content)
    with pytest.raises(ValueError, match=match):
        import_checkpoint(source, tmp_path / "model/best.pt")


def test_model_size_limit_checked_before_pytorch_load(tmp_path, monkeypatch):
    monkeypatch.setattr("facejpeg.daily.MAX_MODEL_BYTES", 10)
    with pytest.raises(ValueError, match="حجمه"):
        verify_checkpoint(SOFTWARE_MODEL)
    archive_path = tmp_path / "model.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("best.pt", b"a" * 11)
    with pytest.raises(ValueError, match="كبير"):
        import_checkpoint(archive_path, tmp_path / "model/best.pt")


def test_discovery_excludes_unselected_candidates_and_filters_shapes_by_default(tmp_path):
    expected = tmp_path / "facejpeg_final_v2_3_1/runs/run/jpeg_augmented/best.pt"
    candidate = tmp_path / "facejpeg_final_v2_3_1/runs/run/candidates/rejected/best.pt"
    for path in [expected, candidate]:
        path.parent.mkdir(parents=True)
        shutil.copyfile(SOFTWARE_MODEL, path)
    assert discover_checkpoints(tmp_path) == []
    assert discover_checkpoints(tmp_path, require_faces=False) == [expected]


def test_single_cell_notebook_matches_script_and_validates_embedded_code():
    nb = nbformat.read(ROOT / "FaceJPEG_Daily_Predict.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    assert len(code) == 1 and code[0] == (ROOT / "daily_predict.py").read_text()
    ast.parse(code[0])
    assert "drive.mount" not in code[0] and "kaggle" not in code[0].lower()


def test_fresh_process_bootstrap_and_next_session_backup_restore(tmp_path):
    # A new process has neither old notebook globals nor the project in sys.path.
    # Internal False mode is solely for this procedural software fixture.
    script = (ROOT / "daily_predict.py").read_text()
    tree = ast.parse(script)
    assert ast.unparse(tree.body[-1]) == "_DAILY_RESULT = _daily_main()"
    body = script[:script.rfind("_DAILY_RESULT = _daily_main()")]
    runner = tmp_path / "cold_session.py"
    runner.write_text(body + '\nresult = _daily_main(require_faces=False)\nassert result is not None\nprint("COLD_BOOT_OK", result["backup"])\n')
    environment = {**os.environ, "FACEJPEG_CHECKPOINT": str(SOFTWARE_MODEL),
                   "FACEJPEG_DAILY_HOME": str(tmp_path / "session_one"), "FACEJPEG_NO_UI": "1"}
    environment.pop("PYTHONPATH", None)
    first = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert first.returncode == 0 and "COLD_BOOT_OK" in first.stdout, first.stdout + first.stderr
    backup = next((tmp_path / "session_one").glob("facejpeg_daily_*/facejpeg_model_backup.zip"))
    # Different runtime filesystem: no source code, sys.path, BEST or model from yesterday.
    environment.update(FACEJPEG_CHECKPOINT=str(backup), FACEJPEG_DAILY_HOME=str(tmp_path / "session_two"))
    second = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert second.returncode == 0 and "COLD_BOOT_OK" in second.stdout, second.stdout + second.stderr
    restored = next((tmp_path / "session_two").glob("facejpeg_daily_*/models/best.pt"))
    assert sha(restored) == sha(SOFTWARE_MODEL)
    assert "ModuleNotFoundError" not in first.stderr + second.stderr
    # Same runtime, new kernel: no explicit upload needed when the model still exists.
    environment["FACEJPEG_CHECKPOINT"] = ""
    third = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    assert third.returncode == 0 and "COLD_BOOT_OK" in third.stdout, third.stdout + third.stderr


def test_public_cold_start_rejects_procedural_model_without_name_error(tmp_path):
    environment = {**os.environ, "FACEJPEG_CHECKPOINT": str(SOFTWARE_MODEL),
                   "FACEJPEG_DAILY_HOME": str(tmp_path), "FACEJPEG_NO_UI": "1"}
    result = subprocess.run([sys.executable, str(ROOT / "daily_predict.py")], cwd=tmp_path,
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0 and "QUICK" in result.stdout, result.stdout + result.stderr
    assert "ModuleNotFoundError" not in result.stderr and "NameError" not in result.stderr
    assert not list(tmp_path.glob("facejpeg_daily_*/models/best.pt"))


def test_colab_upload_backup_and_launch_flow_with_software_fixture(tmp_path):
    script = (ROOT / "daily_predict.py").read_text()
    body = script[:script.rfind("_DAILY_RESULT = _daily_main()")]
    # Simulated Colab bridge: no access to the user's account or public share tunnel.
    bridge = '''
import types
import gradio as gr
colab = types.ModuleType("google.colab")
uploads, downloads, launches = [], [], []
def upload():
    uploads.append(True)
    return {"best.pt": Path(os.environ["SOFTWARE_FIXTURE"]).read_bytes()}
def download(path):
    assert Path(path).is_file()
    downloads.append(path)
colab.files = types.SimpleNamespace(upload=upload, download=download)
sys.modules["google.colab"] = colab
old_launch = gr.Blocks.launch
def fake_launch(self, **kwargs):
    launches.append(kwargs)
    return None
gr.Blocks.launch = fake_launch
result = _daily_main(require_faces=False)
assert result is not None and len(uploads) == len(downloads) == len(launches) == 1
assert launches[0]["share"] is True and launches[0]["max_file_size"] == "15mb"
assert BEST == result["checkpoint"]
tables = [c for c in result["demo"].config["components"] if c["type"] == "dataframe"]
assert len(tables) == 2 and all(c["props"]["visible"] is False for c in tables)
result["demo"].close()
print("COLAB_BRIDGE_OK — software fixture only, no face accuracy claim")
'''
    runner = tmp_path / "simulated_colab.py"
    runner.write_text(body + bridge)
    environment = {**os.environ, "FACEJPEG_CHECKPOINT": "", "FACEJPEG_DAILY_HOME": str(tmp_path),
                   "SOFTWARE_FIXTURE": str(SOFTWARE_MODEL)}
    environment.pop("FACEJPEG_NO_UI", None)
    result = subprocess.run([sys.executable, str(runner)], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0 and "COLAB_BRIDGE_OK" in result.stdout, result.stdout + result.stderr
