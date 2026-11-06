import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .common import (CLASSES, PREPROCESS, atomic_torch_save, build_model, decisions,
                     file_sha256, read_rgb, seed_everything, tensorize, write_json)


class Faces(Dataset):
    def __init__(self, frame, root, config=None, augment=False):
        self.frame, self.root = frame.reset_index(drop=True), Path(root)
        self.config, self.augment = config or PREPROCESS, augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        im = read_rgb(self.root / row.path)
        if self.augment and torch.rand(1).item() < .5:
            from PIL import Image
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return tensorize(im, self.config), torch.tensor(float(row.label))


@torch.inference_mode()
def predict_loader(model, loader, device):
    model.eval()
    labels, scores, losses = [], [], []
    for x, y in loader:
        logits = model(x.to(device)).flatten()
        losses.extend(torch.nn.functional.binary_cross_entropy_with_logits(logits, y.to(device), reduction="none").cpu().tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
        labels.extend(y.int().tolist())
    return np.asarray(labels), np.asarray(scores), float(np.mean(losses))


def select_thresholds(labels, scores):
    """Validation-only utility: +1 correct, -4 false accusation, -1 missed fake, 0 abstain."""
    candidates = []
    for low in np.linspace(.05, .45, 17):
        for high in np.linspace(.55, .95, 17):
            pred = decisions(scores, {"low": float(low), "high": float(high)})
            correct = int(((pred == labels) & (pred != 2)).sum())
            fp = int(((labels == 0) & (pred == 1)).sum())
            fn = int(((labels == 1) & (pred == 0)).sum())
            coverage = float((pred != 2).mean())
            candidates.append({"low": float(low), "high": float(high),
                "utility": (correct - 4 * fp - fn) / len(labels), "coverage": coverage})
    # Utility first, then coverage, then a wider abstention interval to break remaining ties.
    best = max(candidates, key=lambda r: (r["utility"], r["coverage"], r["high"] - r["low"]))
    return {k: best[k] for k in ["low", "high"]}, candidates


def train(prepared, output, epochs=10, batch_size=32, lr=1e-4, quick=False, resume=True,
          pretrained=True, seed=42):
    seed_everything(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    root, out = Path(prepared), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "manifest.csv")
    audit = json.loads((root / "audit.json").read_text())
    for split in ["train", "val", "test"]:
        if set(frame.loc[frame.split == split, "label"]) != {0, 1}:
            raise ValueError(f"Missing class in {split}")
    device = "cuda" if torch.cuda.is_available() and not quick else "cpu"
    last_path, best_path = out / "last.pt", out / "best.pt"
    manifest_hash = file_sha256(root / "manifest.csv")
    previous = torch.load(last_path, map_location="cpu", weights_only=True) if resume and last_path.exists() else None
    if not resume and (best_path.exists() or last_path.exists()):
        raise FileExistsError("Use a new output directory for a new experiment")
    if previous and (previous["manifest_sha256"] != manifest_hash or previous["quick"] != quick or previous["seed"] != seed):
        raise ValueError("Resume requires identical manifest, mode and seed; choose new output")
    model = build_model(pretrained=pretrained and previous is None).to(device)
    if quick:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=.01)
    start, best_loss, stale, history = 0, float("inf"), 0, []
    if previous:
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        start, best_loss = previous["epoch"] + 1, previous["best_val_loss"]
        history, stale = previous["history"], previous["stale"]
        torch.set_rng_state(previous["torch_rng"])
        random.setstate(previous["python_rng"])
        if device == "cuda" and previous.get("cuda_rng"):
            torch.cuda.set_rng_state_all(previous["cuda_rng"])
        if not best_path.exists():
            raise FileNotFoundError("Copy both best.pt and last.pt to resume")
    train_frame, val_frame = frame[frame.split == "train"], frame[frame.split == "val"]
    val_loader = DataLoader(Faces(val_frame, root), batch_size=batch_size, num_workers=0)
    counts = train_frame.label.value_counts()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(counts[0] / counts[1]), device=device))
    base = {"schema_version": 1, "architecture": "mobilenet_v3_small", "classes": CLASSES,
            "preprocess": PREPROCESS, "thresholds": None, "manifest_sha256": manifest_hash,
            "seed": seed, "quick": quick, "data_kind": audit["data_kind"],
            "pretrained": previous["pretrained"] if previous else pretrained,
            "run_label": "SMOKE_ONLY_NOT_A_FACE_DETECTOR" if audit["data_kind"] == "procedural_fixture" else "QUICK_SUBSET" if quick else "FULL_TRAINING"}
    print(f"device={device}; mode={base['run_label']}; train={len(train_frame)} val={len(val_frame)}")
    for epoch in range(start, epochs):
        model.train()
        if quick:
            model.features.eval()  # Preserve pretrained BN statistics during tiny CPU smoke run.
        gen = torch.Generator().manual_seed(seed + epoch)
        loader = DataLoader(Faces(train_frame, root, augment=not quick), batch_size=batch_size,
                            shuffle=True, generator=gen, num_workers=0)
        total = 0.
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)).flatten(), y.to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        _, _, val_loss = predict_loader(model, val_loader, device)
        record = {"epoch": epoch + 1, "train_loss": total / len(train_frame), "val_loss": val_loss}
        history.append(record)
        print(record, flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            atomic_torch_save({**base, "model_state": model.state_dict(), "epoch": epoch,
                               "val_loss": val_loss}, best_path)
        else:
            stale += 1
        atomic_torch_save({**base, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "epoch": epoch, "best_val_loss": best_loss, "stale": stale, "history": history,
            "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}, last_path)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if stale >= 5:
            break
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    y, s, _ = predict_loader(model, val_loader, device)
    thresholds, grid = select_thresholds(y, s)
    best["thresholds"] = thresholds
    best["threshold_selection"] = "validation_only_utility_correct1_falseaccusation-4_missedfake-1_abstain0"
    atomic_torch_save(best, best_path)
    pd.DataFrame(grid).to_csv(out / "validation_threshold_grid.csv", index=False)
    pd.DataFrame({"path": val_frame.path.tolist(), "label": y, "score_fake": s,
                  "decision": decisions(s, thresholds)}).to_csv(out / "validation_predictions.csv", index=False)
    write_json(out / "run_info.json", {k: v for k, v in best.items() if k != "model_state"})
    return best_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()
    train(a.prepared, a.output, a.epochs, a.batch_size, quick=a.quick, resume=not a.no_resume)
