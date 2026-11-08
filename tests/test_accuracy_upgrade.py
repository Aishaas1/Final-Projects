"""Refinement correctness on procedural fixtures; no face-accuracy claim."""
from pathlib import Path
from unittest.mock import Mock

import nbformat
import pytest
import torch

from facejpeg.common import PREPROCESS, build_model, file_sha256, load_bundle, seed_everything
from facejpeg.data import prepare
from facejpeg.train import configure_transfer, make_optimizer, train
from facejpeg.upgrade import improve_accuracy, profiles, select_candidate, validation_report
from scripts.make_demo import make_demo


@pytest.fixture
def initialized(tmp_path):
    root = tmp_path / "prepared"
    prepare(make_demo(tmp_path / "raw", 8), root)
    options = dict(quick=True, pretrained=False, cache_features=True,
                   warmup_epochs=0, finetune_blocks=0, batch_size=4)
    checkpoint = train(root, tmp_path / "initial", epochs=1, **options)
    return root, checkpoint, options


def test_warm_start_resume_is_exact_and_leaves_initial_weights_untouched(tmp_path, initialized):
    root, initial, options = initialized
    before = file_sha256(initial)
    kwargs = {**options, "initial_checkpoint": initial, "lr": 1e-4,
              "feature_cache_directory": tmp_path / "shared_cache"}
    train(root, tmp_path / "resumed", epochs=1, **kwargs)
    train(root, tmp_path / "resumed", epochs=2, **kwargs)
    train(root, tmp_path / "full", epochs=2, **kwargs)
    a = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    b = torch.load(tmp_path / "full/last.pt", weights_only=True)
    assert all(torch.equal(a["model_state"][key], b["model_state"][key]) for key in a["model_state"])
    assert a["training_config"]["initial_checkpoint_sha256"] == before
    assert len(list((tmp_path / "shared_cache/feature_cache").glob("*.pt"))) == 1
    assert file_sha256(initial) == before
    with pytest.raises(ValueError, match="Resume settings changed"):
        train(root, tmp_path / "resumed", epochs=2, **{**kwargs, "initial_checkpoint": None})


def test_warm_start_rejects_different_split(tmp_path, initialized):
    _, initial, options = initialized
    other = tmp_path / "other"
    prepare(make_demo(tmp_path / "other_raw", 9), other)
    with pytest.raises(ValueError, match="identical prepared split"):
        train(other, tmp_path / "rejected", epochs=1, initial_checkpoint=initial, **options)


def test_differential_learning_rates_only_update_selected_blocks():
    seed_everything(42)
    model = build_model(False).eval()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    selected = configure_transfer(model, False, 2, False)
    optimizer, parameters = make_optimizer(model, selected, 1e-3, .1)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-3, 1e-4]
    assert len({id(p) for p in parameters}) == len(parameters)
    x, y = torch.rand(2, 3, 224, 224), torch.tensor([0., 1.])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), y)
    loss.backward(); optimizer.step()
    changed = [name for name, tensor in model.state_dict().items() if not torch.equal(before[name], tensor)]
    allowed = {id(parameter) for block in selected for parameter in block.parameters()}
    allowed |= {id(parameter) for parameter in model.classifier.parameters()}
    assert changed and all(id(parameter) in allowed or torch.equal(before[name], parameter)
                           for name, parameter in model.named_parameters())
    assert any(id(p) in allowed and not torch.equal(before[name], p)
               for name, p in model.named_parameters() if name.startswith("features."))
    compatible, _ = make_optimizer(model, selected, 1e-3)
    assert len(compatible.param_groups) == 1


def test_refinement_validation_never_reads_train_or_test_images(tmp_path, initialized, monkeypatch):
    root, initial, _ = initialized
    import facejpeg.upgrade as upgrade
    original_read = upgrade.read_rgb
    read_paths = []

    def guarded(path):
        assert "/val/" in str(path) and "/test/" not in str(path) and "/train/" not in str(path)
        read_paths.append(str(path))
        return original_read(path)

    monkeypatch.setattr(upgrade, "read_rgb", guarded)
    report = validation_report(root, initial, tmp_path / "validation", batch_size=4)
    assert read_paths and 0 <= report["mean_balanced_accuracy"] <= 1
    import pandas as pd
    predictions = pd.read_csv(tmp_path / "validation/validation_predictions.csv")
    assert set(predictions.quality) == {"original", "90", "70", "50", "30"}
    assert set(predictions.path) == {str(Path(p).relative_to(root)) for p in read_paths}


@pytest.mark.parametrize("mean,original,expected", [(.79, .85, "old"), (.8, .9, "old"),
                                                    (.81, .81, "old"), (.81, .82, "new")])
def test_selector_requires_strict_mean_gain_and_original_floor(mean, original, expected):
    reports = {"old": {"mean_balanced_accuracy": .8, "original_balanced_accuracy": .82},
               "new": {"mean_balanced_accuracy": mean, "original_balanced_accuracy": original}}
    assert select_candidate(reports, "old") == expected


def test_upgrade_preserves_original_and_locks_validation_selection(tmp_path, initialized, monkeypatch):
    root, initial, _ = initialized
    before = file_sha256(initial)
    out = tmp_path / "upgrade"
    best, selection = improve_accuracy(root, {"current": initial}, out, max_epochs=2, software_test=True, incumbent_name="current")
    assert best.is_file() and file_sha256(initial) == before
    assert not selection["test_used_for_selection"] and not selection["test_accuracy_gain_measured"]
    assert not selection["validation_gain_on_research_faces"]
    model, bundle = load_bundle(best)
    assert bundle["data_kind"] == "procedural_fixture" and bundle["preprocess"] == PREPROCESS
    del model
    import facejpeg.upgrade as upgrade
    no_retrain = Mock(side_effect=AssertionError("A completed refinement must be reused"))
    monkeypatch.setattr(upgrade, "train", no_retrain)
    repeat, repeated = improve_accuracy(root, {"current": initial}, out, max_epochs=2, software_test=True, incumbent_name="current")
    assert repeat == best and repeated == selection
    no_retrain.assert_not_called()
    with pytest.raises(ValueError, match="وجوه"):
        improve_accuracy(root, {"current": initial}, tmp_path / "not_faces")
    assert (out / "selected_protocol.json").is_file()


def test_cpu_and_gpu_profiles_do_real_feature_finetuning():
    cpu, gpu = profiles(False)[1], profiles(True)[1]
    assert cpu["finetune_blocks"] == 2 and not cpu["cache_features"]
    assert gpu["finetune_blocks"] == 4 and not gpu["cache_features"]
    assert cpu["backbone_lr_factor"] == gpu["backbone_lr_factor"] == .1


def test_upgrade_cell_without_working_session_does_not_change_best(capsys):
    source = Path("accuracy_upgrade_cell.py").read_text()
    scope = {"BEST": "keep-current-model.pt"}
    exec(compile(source, "accuracy_upgrade_cell.py", "exec"), scope)
    assert scope["BEST"] == "keep-current-model.pt"
    assert "داخل دفتر FaceJPEG" in capsys.readouterr().out
    notebook = nbformat.read("FaceJPEG_Accuracy_Upgrade.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert len([c for c in notebook.cells if c.cell_type == "code"]) == 1
