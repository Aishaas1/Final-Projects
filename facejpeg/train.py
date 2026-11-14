import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .augmentation import jpeg_settings, jpeg_training_view
from .common import (CLASSES, PREPROCESS, atomic_torch_save, build_model, decisions,
                     file_sha256, load_bundle, read_rgb, seed_everything, tensorize, write_json)
from .data import validate_manifest
from .features import FeatureRows, cached_features
from .runtime import Monitor, choose_batch_size, hardware_info, loader_options


class Faces(Dataset):
    def __init__(self, frame, root, config=None, augment=False, jpeg_qualities=(), jpeg_probability=.5):
        self.frame, self.root = frame.reset_index(drop=True), Path(root)
        self.config, self.augment = config or PREPROCESS, augment
        self.jpeg_qualities, self.jpeg_probability = jpeg_settings(jpeg_qualities, jpeg_probability)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        im = read_rgb(self.root / row.path)
        if self.jpeg_qualities and torch.rand(1).item() < self.jpeg_probability:
            quality = self.jpeg_qualities[int(torch.randint(len(self.jpeg_qualities), (1,)).item())]
            im = jpeg_training_view(im, quality, self.config)
        if self.augment and torch.rand(1).item() < .5:
            from PIL import Image
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return tensorize(im, self.config), torch.tensor(float(row.label))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.inference_mode()
def predict_loader(model, loader, device, monitor=None):
    model.eval()
    labels, scores, total_loss, n = [], [], 0., 0
    for x, y in tqdm(loader, desc="Validation batches", leave=False):
        logits = model(x.to(device, non_blocking=True)).flatten()
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y.to(device, non_blocking=True), reduction="sum").item()
        n += len(y)
        scores.extend(logits.sigmoid().cpu().tolist())
        labels.extend(y.int().tolist())
        if monitor:
            monitor.advance(len(y), "validation batch completed")
    if not n:
        raise ValueError("Validation loader is empty")
    return np.asarray(labels), np.asarray(scores), total_loss / n


def select_thresholds(labels, scores):
    """Validation-only: +1 correct, -4 false accusation, -1 missed fake, 0 abstain."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not len(labels) or set(labels) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires finite validation scores and both classes")
    candidates = []
    for low in np.linspace(.05, .45, 17):
        for high in np.linspace(.55, .95, 17):
            pred = decisions(scores, {"low": float(low), "high": float(high)})
            correct = int(((pred == labels) & (pred != 2)).sum())
            fp = int(((labels == 0) & (pred == 1)).sum())
            fn = int(((labels == 1) & (pred == 0)).sum())
            candidates.append({"low": float(low), "high": float(high),
                "utility": (correct - 4 * fp - fn) / len(labels), "coverage": float((pred != 2).mean())})
    best = max(candidates, key=lambda r: (r["utility"], r["coverage"], r["high"] - r["low"]))
    return {k: best[k] for k in ["low", "high"]}, candidates


def configure_transfer(model, quick, blocks, warmup):
    for p in model.features.parameters():
        p.requires_grad = False
    selected = list(model.features) if blocks == -1 else list(model.features)[-blocks:] if blocks > 0 else []
    if not quick and not warmup:
        for block in selected:
            for p in block.parameters():
                p.requires_grad = True
    return selected


def make_optimizer(model, selected, lr, backbone_lr_factor=1., quick=False, legacy=False):
    head = list(model.classifier.parameters())
    backbone = [] if quick else [p for block in selected for p in block.parameters()]
    parameters = head + backbone
    if legacy and not quick:
        parameters = list(model.parameters())
    if backbone and backbone_lr_factor != 1. and not legacy:
        groups = [{"params": head, "lr": lr}, {"params": backbone, "lr": lr * backbone_lr_factor}]
        optimizer = torch.optim.AdamW(groups, lr=lr, weight_decay=.01)
    else:
        # Preserve the single parameter-group layout of existing checkpoints.
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=.01)
    return optimizer, parameters


def save_learning_curves(history, out, run_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame.epoch, frame.train_loss, "o-", label="Train (class-weighted BCE)")
    axes[0].plot(frame.epoch, frame.val_loss, "o-", label="Validation (unweighted BCE)")
    for col, label in [("train_accuracy_at_0_5", "Train online @0.5"), ("val_accuracy_at_0_5", "Validation @0.5")]:
        if col in frame:
            axes[1].plot(frame.epoch, frame[col], "o-", label=label)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=.3)
    axes[0].set_ylabel("Loss"); axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1.05)
    fig.suptitle(run_label); fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=140); plt.close(fig)


def train(prepared, output, epochs=10, batch_size=None, lr=1e-4, quick=False, resume=True,
          pretrained=True, seed=42, warmup_epochs=1, finetune_blocks=3, use_amp=True,
          workers=None, allow_cpu_real=False, cache_features=False, use_scheduler=False,
          early_stopping_patience=5, jpeg_qualities=(), jpeg_probability=.5,
          initial_checkpoint=None, backbone_lr_factor=1., feature_cache_directory=None):
    with Monitor(output) as monitor:
        try:
            return _train(prepared, output, epochs, batch_size, lr, quick, resume, pretrained,
                          seed, warmup_epochs, finetune_blocks, use_amp, workers, allow_cpu_real,
                          cache_features, use_scheduler, early_stopping_patience,
                          jpeg_qualities, jpeg_probability, initial_checkpoint, backbone_lr_factor,
                          feature_cache_directory, monitor)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("نفدت ذاكرة GPU. اختر BATCH_SIZE أصغر واسم تجربة جديد، وأغلق النماذج الأخرى. لم يُعد التدريب تلقائيًا على CPU.") from exc


def _train(prepared, output, epochs, batch_size, lr, quick, resume, pretrained, seed,
           warmup_epochs, finetune_blocks, use_amp, workers, allow_cpu_real,
           cache_features, use_scheduler, early_stopping_patience, jpeg_qualities, jpeg_probability,
           initial_checkpoint, backbone_lr_factor, feature_cache_directory, monitor):
    jpeg_qualities, jpeg_probability = jpeg_settings(jpeg_qualities, jpeg_probability)
    if epochs < 1 or lr <= 0 or finetune_blocks < -1 or warmup_epochs < 0 or early_stopping_patience < 1:
        raise ValueError("Invalid epochs/lr/transfer-learning configuration")
    if not np.isfinite(backbone_lr_factor) or not 0 < backbone_lr_factor <= 1:
        raise ValueError("backbone_lr_factor must be in (0,1]")
    seed_everything(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root, out = Path(prepared), Path(output)
    monitor.stage("التحقق من التقسيم وبيئة التشغيل")
    frame = validate_manifest(root, verify_files=True)
    audit = json.loads((root / "audit.json").read_text())
    if not quick and audit["data_kind"] == "procedural_fixture":
        raise ValueError("Real Training لا يقبل بيانات fixture. اختر بيانات وجوه أو Quick Test.")
    device = "cuda" if torch.cuda.is_available() and not quick and not cache_features else "cpu"
    if cache_features and (finetune_blocks != 0 or warmup_epochs != 0):
        raise ValueError("Cached CPU training requires finetune_blocks=0 and warmup_epochs=0")
    if not quick and device == "cpu" and not allow_cpu_real:
        raise RuntimeError("Real Training يحتاج GPU افتراضيًا. فعّل GPU في Colab، أو اختر Quick Test. للسماح الواعي بتدريب CPU البطيء مرّر allow_cpu_real=True.")
    batch_size = batch_size or choose_batch_size(quick)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    hw = hardware_info()
    write_json(out / "hardware.json", hw)
    monitor.event(f"device={device}; batch_size={batch_size}; GPU={hw.get('gpu_name', 'none')}")
    last_path, best_path = out / "last.pt", out / "best.pt"
    manifest_hash = file_sha256(root / "manifest.csv")
    initial_model, initial_info, initial_hash = None, None, None
    if initial_checkpoint is not None:
        initial_hash = file_sha256(initial_checkpoint)
        initial_model, initial_info = load_bundle(initial_checkpoint, "cpu")
        if initial_info["manifest_sha256"] != manifest_hash or initial_info["data_kind"] != audit["data_kind"]:
            raise ValueError("Warm start requires the identical prepared split and dataset kind")
        if initial_info["preprocess"] != PREPROCESS:
            raise ValueError("Warm start preprocessing differs from this training recipe")
    previous = torch.load(last_path, map_location="cpu", weights_only=True) if resume and last_path.exists() else None
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError("Use a new output directory for a new experiment")
    if previous and (previous["manifest_sha256"] != manifest_hash or previous["quick"] != quick or previous["seed"] != seed):
        raise ValueError("Resume requires identical manifest, mode and seed; choose new output")
    legacy = bool(previous and "training_config" not in previous)
    if legacy:
        if cache_features or use_scheduler or jpeg_qualities or initial_checkpoint or backbone_lr_factor != 1.:
            raise ValueError("Legacy checkpoint has a different strategy; choose a new experiment for cached CPU training")
        # Keep old checkpoint optimizer layout; never silently discard its optimizer.
        warmup_epochs, finetune_blocks = 0, -1
        monitor.event("استكمال checkpoint قديم بإعدادات التجميد الأصلية؛ لا تُقارن الأزمنة بتشغيل جديد.")
    config = {"batch_size": batch_size, "lr": lr, "warmup_epochs": 0 if quick else warmup_epochs,
              "finetune_blocks": 0 if quick else finetune_blocks, "amp": bool(use_amp and device == "cuda"),
              "workers": loader_options(device, quick, workers)["num_workers"],
              "cache_features": bool(cache_features), "use_scheduler": bool(use_scheduler),
              "early_stopping_patience": early_stopping_patience,
              "jpeg_qualities": list(jpeg_qualities), "jpeg_probability": jpeg_probability,
              "initial_checkpoint_sha256": initial_hash, "backbone_lr_factor": float(backbone_lr_factor)}
    defaults = {"cache_features": False, "use_scheduler": False, "early_stopping_patience": 5,
                "jpeg_qualities": [], "jpeg_probability": 0.,
                "initial_checkpoint_sha256": None, "backbone_lr_factor": 1.}
    if previous and previous.get("training_config") and {**defaults, **previous["training_config"]} != config:
        raise ValueError("Resume settings changed (batch/lr/AMP/transfer/workers); restore original config or use a new experiment")
    monitor.stage("تحميل MobileNetV3 والأوزان الرسمية")
    model = (initial_model if initial_model is not None else
             build_model(pretrained=pretrained and previous is None)).to(device)
    selected = configure_transfer(model, quick, config["finetune_blocks"], warmup=False)
    optimizer, parameters = make_optimizer(model, selected, lr, backbone_lr_factor, quick, legacy)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5,
                 patience=2, min_lr=1e-6) if use_scheduler else None)
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp"])
    start, best_loss, stale, history = 0, float("inf"), 0, []
    if previous:
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        if scheduler:
            if not previous.get("scheduler_state"):
                raise ValueError("Scheduler state missing; start a new experiment instead of guessing it")
            scheduler.load_state_dict(previous["scheduler_state"])
        if previous.get("scaler_state"):
            scaler.load_state_dict(previous["scaler_state"])
        start, best_loss = previous["epoch"] + 1, previous["best_val_loss"]
        history, stale = previous["history"], previous["stale"]
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        if device == "cuda" and previous.get("cuda_rng"):
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        if not best_path.exists():
            raise FileNotFoundError("Copy both best.pt and last.pt to resume")
    train_frame, val_frame = frame[frame.split == "train"], frame[frame.split == "val"]
    options = loader_options(device, quick, workers)
    forward_model = model
    train_dataset = Faces(train_frame, root, augment=not quick,
                          jpeg_qualities=jpeg_qualities, jpeg_probability=jpeg_probability)
    val_dataset = Faces(val_frame, root)
    if cache_features:
        cache, cache_path = cached_features(model, frame, root,
                                            out if feature_cache_directory is None else feature_cache_directory,
                                            PREPROCESS, monitor,
                                            batch_size=min(batch_size, 16), jpeg_qualities=jpeg_qualities)
        forward_model = model.classifier
        train_dataset = FeatureRows(cache["train_original"], cache["train_labels"],
                                    cache["train_flipped"], augment=not quick,
                                    jpeg_original=cache.get("train_jpeg_original"),
                                    jpeg_flipped=cache.get("train_jpeg_flipped"),
                                    jpeg_probability=jpeg_probability)
        val_dataset = FeatureRows(cache["val_original"], cache["val_labels"])
        write_json(out / "feature_cache_info.json", {"file": cache_path.name,
            "metadata": cache["metadata"], "test_images_cached": False,
            "backbone_frozen": True, "feature_dimension": int(cache["train_original"].shape[1]),
            "cache_reused": cache["cache_reused"],
            "extraction_seconds_when_built": cache.get("extraction_seconds_when_built")})
    val_loader = DataLoader(val_dataset, batch_size=batch_size, **options,
                            generator=torch.Generator().manual_seed(seed + 100000), worker_init_fn=seed_worker)
    counts = train_frame.label.value_counts()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(counts[0] / counts[1]), device=device))
    base = {"schema_version": 1, "architecture": "mobilenet_v3_small", "classes": CLASSES,
            "preprocess": PREPROCESS, "thresholds": None, "manifest_sha256": manifest_hash,
            "seed": seed, "quick": quick, "data_kind": audit["data_kind"], "training_config": config,
            "pretrained": previous["pretrained"] if previous else initial_info["pretrained"] if initial_info else pretrained,
            "run_label": "SMOKE_ONLY_NOT_A_FACE_DETECTOR" if audit["data_kind"] == "procedural_fixture" else
                         "QUICK_SUBSET" if quick else "REAL_CPU_FROZEN_BACKBONE" if cache_features else "REAL_TRAINING"}
    for epoch in range(start, epochs):
        if stale >= early_stopping_patience:
            monitor.event("تحقق الإيقاف المبكر سابقًا؛ نعيد تحميل الأفضل دون استئناف غير مقصود")
            break
        tick = time.monotonic()
        warming = epoch < config["warmup_epochs"]
        configure_transfer(model, quick, config["finetune_blocks"], warmup=warming)
        model.train()
        # Frozen backbone/blocks keep pretrained BN statistics; unfreeze only selected blocks.
        model.features.eval()
        if not quick and not warming:
            for block in selected:
                block.train()
        gen = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, generator=gen, worker_init_fn=seed_worker, **options)
        monitor.stage(f"تدريب Epoch {epoch+1}/{epochs}" + (" — warmup head" if warming else ""), len(train_frame))
        total, correct, seen = 0., 0, 0
        bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for batch, (x, y) in enumerate(bar, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=config["amp"]):
                logits = forward_model(x).flatten()
                loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss; no invalid checkpoint saved")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(y)
            correct += int(((logits.detach() >= 0).long() == y.long()).sum().item())
            seen += len(y)
            bar.set_postfix(loss=f"{total/seen:.4f}", accuracy=f"{correct/seen:.3f}")
            monitor.advance(len(y), f"Epoch {epoch+1} batch {batch}/{len(loader)} saved in memory")
        monitor.stage(f"تحقق Epoch {epoch+1}/{epochs}", len(val_frame))
        yval, sval, val_loss = predict_loader(forward_model, val_loader, device, monitor)
        if not np.isfinite(val_loss) or not np.isfinite(sval).all():
            raise FloatingPointError("Non-finite validation result; checkpoint was not saved")
        seconds = time.monotonic() - tick
        record = {"epoch": epoch + 1, "train_loss": total / seen, "val_loss": val_loss,
                  "train_accuracy_at_0_5": correct / seen,
                  "val_accuracy_at_0_5": float(((sval >= .5).astype(int) == yval).mean()),
                  "epoch_seconds": seconds, "train_images_per_second": seen / seconds,
                  "learning_rate_used": float(optimizer.param_groups[0]["lr"]),
                  "eta_remaining_epochs_seconds": seconds * (epochs - epoch - 1)}
        if scheduler:
            scheduler.step(val_loss)
        record["learning_rate_next"] = float(optimizer.param_groups[0]["lr"])
        history.append(record)
        monitor.stage("حفظ checkpoint ونتائج الحقبة")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            atomic_torch_save({**base, "model_state": model.state_dict(), "epoch": epoch,
                               "val_loss": val_loss}, best_path)
        else:
            stale += 1
        atomic_torch_save({**base, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(), "scheduler_state": scheduler.state_dict() if scheduler else {},
            "epoch": epoch, "best_val_loss": best_loss,
            "stale": stale, "history": history, "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}, last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        monitor.event(f"checkpoint حقبة {epoch+1} محفوظ | loss={total/seen:.4f} val_loss={val_loss:.4f} "
                      f"accuracy={correct/seen:.3f} val_accuracy={record['val_accuracy_at_0_5']:.3f} "
                      f"epoch={seconds:.1f}s ETA≈{record['eta_remaining_epochs_seconds']:.1f}s")
    monitor.stage("اختيار الحدود من التحقق فقط", len(val_frame))
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    y, s, _ = predict_loader(forward_model, val_loader, device, monitor)
    thresholds, grid = select_thresholds(y, s)
    best["thresholds"] = thresholds
    best["threshold_selection"] = "validation_only_utility_correct1_falseaccusation-4_missedfake-1_abstain0"
    atomic_torch_save(best, best_path)
    pd.DataFrame(grid).to_csv(out / "validation_threshold_grid.csv", index=False)
    pd.DataFrame({"path": val_frame.path.tolist(), "label": y, "score_fake": s,
                  "decision": decisions(s, thresholds)}).to_csv(out / "validation_predictions.csv", index=False)
    write_json(out / "run_info.json", {k: v for k, v in best.items() if k != "model_state"})
    save_learning_curves(history, out, best["run_label"])
    epoch_times = [r["epoch_seconds"] for r in history if "epoch_seconds" in r]
    write_json(out / "timing.json", {"device": device, "run_label": best["run_label"],
        "measured_epoch_seconds": epoch_times, "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else None,
        "scope": "training plus validation; excludes download/audit/checkpoint writes" +
                 (" and one-time backbone feature extraction; see feature_cache_info.json" if cache_features else ""),
        "cached_feature_training": bool(cache_features),
        "gpu_estimate": "not measured" if device == "cpu" else "measured on this GPU only"})
    monitor.event(f"أفضل نموذج وحدود التحقق جاهزة: {best_path}")
    return best_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--allow-cpu-real", action="store_true")
    p.add_argument("--cache-features", action="store_true")
    p.add_argument("--scheduler", action="store_true")
    p.add_argument("--jpeg-qualities", type=int, nargs="*", default=[])
    p.add_argument("--jpeg-probability", type=float, default=.5)
    a = p.parse_args()
    train(a.prepared, a.output, a.epochs, a.batch_size, quick=a.quick,
          resume=not a.no_resume, allow_cpu_real=a.allow_cpu_real or a.cache_features, cache_features=a.cache_features,
          finetune_blocks=0 if a.cache_features else 3, warmup_epochs=0 if a.cache_features else 1,
          use_scheduler=a.scheduler, jpeg_qualities=a.jpeg_qualities, jpeg_probability=a.jpeg_probability)
