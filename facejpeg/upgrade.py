"""Warm-start accuracy refinement selected only on validation images.

The incumbent is retained unless mean validation balanced accuracy improves
without reducing balanced accuracy on original validation images. No test
predictions or changes to displayed probabilities are made here.
"""
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import torch
from tqdm.auto import tqdm

from .common import canonical, decisions, file_sha256, jpeg_copy, load_bundle, read_rgb, tensorize, write_json
from .data import validate_manifest
from .evaluate import summary_metrics
from .train import train

VALIDATION_QUALITIES = ("original", 90, 70, 50, 30)


def _copy_atomic(source, target):
    temp = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(target)


def _json_atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def profiles(gpu=False, software_test=False):
    head = {"name": "head_refinement", "lr": 1e-4, "finetune_blocks": 0,
            "cache_features": True, "backbone_lr_factor": 1., "batch_size": 32}
    if software_test:
        return [head]
    fine = {"name": "feature_finetuning", "lr": 1e-4,
            "finetune_blocks": 4 if gpu else 2, "cache_features": False,
            "backbone_lr_factor": .1, "batch_size": 16 if gpu else 8}
    return [head, fine]


def validation_report(prepared, checkpoint, output, device="cpu", batch_size=32):
    """Score only val rows; JPEG views come independently from each original."""
    root, out = Path(prepared), Path(output)
    frame = validate_manifest(root).query("split == 'val'").reset_index(drop=True)
    model, bundle = load_bundle(checkpoint, device)
    if bundle["manifest_sha256"] != file_sha256(root / "manifest.csv"):
        raise ValueError("Validation data differs from the checkpoint split")
    rows, predictions = [], []
    with torch.inference_mode():
        for quality in VALIDATION_QUALITIES:
            scores = []
            for start in tqdm(range(0, len(frame), batch_size), desc=f"Validation Q={quality}", leave=False):
                images = [canonical(read_rgb(root / p), bundle["preprocess"])
                          for p in frame.path.iloc[start:start + batch_size]]
                if quality != "original":
                    images = [jpeg_copy(image, quality) for image in images]
                x = torch.stack([tensorize(image, bundle["preprocess"]) for image in images]).to(device)
                scores.extend(model(x).flatten().sigmoid().cpu().tolist())
            y = frame.label.to_numpy()
            metrics = summary_metrics(y, scores, decisions(scores, bundle["thresholds"]))
            balanced = float(balanced_accuracy_score(y, np.asarray(scores) >= .5))
            rows.append({"quality": str(quality), "n": len(y), "balanced_accuracy_at_0_5": balanced, **metrics})
            predictions.append(pd.DataFrame({"path": frame.path, "label": y, "quality": str(quality), "score_fake": scores}))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "validation_by_quality.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out / "validation_predictions.csv", index=False)
    return {"mean_balanced_accuracy": float(np.mean([r["balanced_accuracy_at_0_5"] for r in rows])),
            "original_balanced_accuracy": rows[0]["balanced_accuracy_at_0_5"],
            "original_binary_accuracy": rows[0]["binary_accuracy_at_0_5_all"],
            "original_roc_auc": rows[0]["roc_auc_all"],
            "mean_abstain_rate": float(np.mean([r["abstain_rate_all"] for r in rows])),
            "mean_false_accusation_rate": float(np.mean([r["false_accusation_rate_all_real"] for r in rows])),
            "mean_fake_detection_rate": float(np.mean([r["fake_detection_rate_all_fake"] for r in rows]))}


def select_candidate(reports, incumbent):
    """Strict validation improvement, with the incumbent-original score as floor."""
    if not incumbent or incumbent not in reports:
        raise ValueError("A valid incumbent validation report is required")
    selected = incumbent
    for name, report in reports.items():
        if (report["mean_balanced_accuracy"] > reports[selected]["mean_balanced_accuracy"] + 1e-12
                and report["original_balanced_accuracy"] + 1e-12 >= reports[incumbent]["original_balanced_accuracy"]):
            selected = name
    return selected


def improve_accuracy(prepared, checkpoints, output, max_epochs=8,
                     software_test=False, pretrained=False, incumbent_name=None):
    """Refine saved models without touching their files or inspecting test results."""
    root, out = Path(prepared), Path(output)
    validate_manifest(root)
    audit = json.loads((root / "audit.json").read_text())
    if audit["data_kind"] != "research_faces" and not software_test:
        raise ValueError("تحسين كشف الوجوه يحتاج بيانات وأوزان وجوه؛ لا يقبل أوزان QUICK الهندسية.")
    if max_epochs < 1 or not checkpoints:
        raise ValueError("Provide trained checkpoints and a positive max_epochs")
    gpu = torch.cuda.is_available() and not software_test
    device = "cuda" if gpu else "cpu"
    torch.set_num_threads(min(4, torch.get_num_threads()))
    recipes = profiles(gpu, software_test)
    sources = {str(name): Path(path).resolve() for name, path in checkpoints.items()}
    reserved = {r["name"] for r in recipes}
    if any(not name or Path(name).name != name or name in reserved for name in sources):
        raise ValueError("Checkpoint names must be simple names distinct from refinement profiles")
    if incumbent_name is not None and incumbent_name not in sources:
        raise ValueError("The incumbent must name one of the provided checkpoints")
    plan = {"schema": 1, "manifest_sha256": file_sha256(root / "manifest.csv"),
            "source_sha256": {name: file_sha256(path) for name, path in sources.items()},
            "max_epochs": max_epochs, "gpu": gpu, "software_test": software_test, "incumbent_name": incumbent_name,
            "profiles": recipes, "validation_qualities": list(VALIDATION_QUALITIES),
            "selection": "strict mean validation balanced accuracy gain; original validation score cannot decrease",
            "test_used_for_selection": False}
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "upgrade_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("خطة التحسين تغيّرت؛ اختر مجلد تحسين جديدًا، ولا تستبدل التجربة السابقة.")
    _json_atomic(plan_path, plan)
    selection_path = out / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        best = out / "best.pt"
        if not best.is_file() or file_sha256(best) != selection["selected_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after validation selection")
        return best, selection
    reports = {}
    for name, path in sources.items():
        model, bundle = load_bundle(path, "cpu")
        del model
        if bundle["data_kind"] != audit["data_kind"] or bundle["manifest_sha256"] != plan["manifest_sha256"]:
            raise ValueError("النموذج والبيانات يجب أن يخصّا التقسيم ونوع البيانات نفسيهما.")
        print("قياس النموذج الحالي على التحقق فقط:", name, flush=True)
        reports[name] = validation_report(root, path, out / "validation" / name, device)
    incumbent = incumbent_name or max(sources, key=lambda name: (reports[name]["mean_balanced_accuracy"],
                                                               reports[name]["original_balanced_accuracy"]))
    anchor = out / "initial_model.pt"
    if anchor.exists() and file_sha256(anchor) != file_sha256(sources[incumbent]):
        raise ValueError("Initial model changed during a partial upgrade")
    _copy_atomic(sources[incumbent], anchor)
    model, anchor_info = load_bundle(anchor, "cpu")
    del model
    augmentation = anchor_info.get("training_config", {})
    all_paths = dict(sources)
    for recipe in recipes:
        options = {k: v for k, v in recipe.items() if k != "name"}
        name = recipe["name"]
        print("بدء تحسين فعلي للأوزان:", name, "device:", "cpu" if options["cache_features"] else device, flush=True)
        path = train(root, out / "candidates" / name, epochs=max_epochs, quick=software_test,
                     pretrained=pretrained, initial_checkpoint=anchor, allow_cpu_real=True,
                     workers=0, warmup_epochs=0, use_scheduler=True, early_stopping_patience=3,
                     jpeg_qualities=augmentation.get("jpeg_qualities", []),
                     jpeg_probability=augmentation.get("jpeg_probability", 0.),
                     feature_cache_directory=out / "shared_features", **options)
        all_paths[name] = path
        reports[name] = validation_report(root, path, out / "validation" / name, device)
    selected = select_candidate(reports, incumbent)
    best = out / "best.pt"
    _copy_atomic(all_paths[selected], best)
    selection = {"incumbent": incumbent, "selected": selected, "reports": reports,
                 "improved_on_validation": selected != incumbent,
                 "selected_checkpoint_sha256": file_sha256(best),
                 "manifest_sha256": plan["manifest_sha256"], "test_used_for_selection": False,
                 "test_accuracy_gain_measured": False,
                 "validation_gain_on_research_faces": selected != incumbent and audit["data_kind"] == "research_faces",
                 "decision_thresholds": "preserved from validation-calibrated selected checkpoint; not edited to increase scores"}
    pd.DataFrame.from_dict(reports, orient="index").rename_axis("model").to_csv(out / "validation_comparison.csv")
    _json_atomic(selection_path, selection)
    _json_atomic(out / "selected_protocol.json", {"checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "selection_data": "validation only",
        "test_status": "not evaluated by the upgrade; previous user test was already inspected"})
    return best, selection
