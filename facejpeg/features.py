"""Frozen-backbone feature cache for genuine CPU transfer learning.

Only train/validation images are cached. Test images stay outside optimization.
"""
import hashlib
import json
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .augmentation import jpeg_settings, jpeg_training_view
from .common import CLASSES, atomic_torch_save, file_sha256, read_rgb, tensorize


class FeatureRows(Dataset):
    def __init__(self, original, labels, flipped=None, augment=False,
                 jpeg_original=None, jpeg_flipped=None, jpeg_probability=.5):
        self.original, self.labels = original, labels.float()
        self.flipped, self.augment = flipped, augment
        self.jpeg_original, self.jpeg_flipped = jpeg_original, jpeg_flipped
        self.jpeg_probability = jpeg_probability

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        flip = self.augment and self.flipped is not None and torch.rand(1).item() < .5
        if self.jpeg_original is not None and torch.rand(1).item() < self.jpeg_probability:
            quality = int(torch.randint(len(self.jpeg_original), (1,)).item())
            views = self.jpeg_flipped if flip else self.jpeg_original
            return views[quality, i], self.labels[i]
        return (self.flipped[i] if flip else self.original[i]), self.labels[i]


class _Images(Dataset):
    def __init__(self, frame, root, preprocess, quality=None):
        self.paths = frame.path.tolist()
        self.root, self.preprocess = Path(root), preprocess
        self.quality = quality

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = read_rgb(self.root / self.paths[i])
        if self.quality is not None:
            image = jpeg_training_view(image, self.quality, self.preprocess)
        return tensorize(image, self.preprocess)


def backbone_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.features.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def cached_features(model, frame, root, output, preprocess, monitor, batch_size=16, jpeg_qualities=()):
    jpeg_qualities, _ = jpeg_settings(jpeg_qualities)
    if any(p.requires_grad for p in model.features.parameters()):
        raise ValueError("Feature caching requires a completely frozen backbone")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("This feature cache is the CPU training path")
    root, output = Path(root), Path(output)
    selected = frame[frame.split.isin(["train", "val"])].copy()
    if selected.empty or set(selected.split) != {"train", "val"}:
        raise ValueError("Feature cache needs train and validation rows")
    # Prepared pixels may change without editing the manifest: verify their content too.
    monitor.stage("التحقق من بصمات صور ميزات CPU", len(selected))
    image_hash = hashlib.sha256()
    for row in tqdm(selected.itertuples(index=False), total=len(selected), desc="Verify cache images", unit="image"):
        image_hash.update(row.path.encode())
        image_hash.update(file_sha256(root / row.path).encode())
        monitor.advance(message=f"verified feature input {row.path}")
    metadata = {"cache_schema": 1, "classes": CLASSES,
                "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__),
                "manifest_sha256": file_sha256(root / "manifest.csv"),
                "prepared_pixels_sha256": image_hash.hexdigest(),
                "backbone_sha256": backbone_digest(model), "preprocess": preprocess,
                "train_paths": frame.loc[frame.split == "train", "path"].tolist(),
                "val_paths": frame.loc[frame.split == "val", "path"].tolist(),
                "augmentation": "original or horizontal flip; training only"}
    if jpeg_qualities:
        metadata.update(cache_schema=2, jpeg_qualities=list(jpeg_qualities),
                        augmentation="original or independent JPEG; optional horizontal flip; training only")
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache_dir = output / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if cache.get("metadata") != metadata:
            raise ValueError("Feature cache metadata invalid; choose a new experiment directory")
        monitor.event("أعيد استخدام ميزات CPU المطابقة للأوزان والصور؛ لم نكرر استخراجها")
        cache["cache_reused"] = True
        return cache, path
    model.eval()
    tick = time.monotonic()
    cache = {"metadata": metadata, "cache_reused": False}
    for split in ["train", "val"]:
        part = frame[frame.split == split].reset_index(drop=True)
        original, flipped = [], []
        monitor.stage(f"استخراج ميزات CPU مرة واحدة: {split}", len(part))
        loader = DataLoader(_Images(part, root, preprocess), batch_size=min(batch_size, 32),
                            num_workers=0, generator=torch.Generator().manual_seed(12345))
        with torch.inference_mode():
            for x in tqdm(loader, desc=f"Extract {split} features", unit="batch"):
                original.append(model.avgpool(model.features(x)).flatten(1).cpu())
                if split == "train":
                    flipped.append(model.avgpool(model.features(x.flip(-1))).flatten(1).cpu())
                monitor.advance(len(x), f"extracted {split} batch")
        # Concatenate OUTSIDE inference_mode: classifier backward must save these inputs.
        cache[f"{split}_original"] = torch.cat(original).clone()
        cache[f"{split}_labels"] = torch.tensor(part.label.tolist(), dtype=torch.float32)
        if flipped:
            cache["train_flipped"] = torch.cat(flipped).clone()
    if jpeg_qualities:
        # Validation stays original; never build or read test views here.
        part = frame[frame.split == "train"].reset_index(drop=True)
        jpeg_original, jpeg_flipped = [], []
        for quality in jpeg_qualities:
            original, flipped = [], []
            monitor.stage(f"ميزات JPEG للتدريب فقط — جودة {quality}", len(part))
            loader = DataLoader(_Images(part, root, preprocess, quality), batch_size=min(batch_size, 32),
                                num_workers=0, generator=torch.Generator().manual_seed(12345))
            with torch.inference_mode():
                for x in tqdm(loader, desc=f"Train JPEG {quality}", unit="batch"):
                    original.append(model.avgpool(model.features(x)).flatten(1).cpu())
                    flipped.append(model.avgpool(model.features(x.flip(-1))).flatten(1).cpu())
                    monitor.advance(len(x), f"extracted train JPEG {quality}")
            jpeg_original.append(torch.cat(original).clone())
            jpeg_flipped.append(torch.cat(flipped).clone())
        cache["train_jpeg_original"] = torch.stack(jpeg_original)
        cache["train_jpeg_flipped"] = torch.stack(jpeg_flipped)
    cache["extraction_seconds_when_built"] = time.monotonic() - tick
    atomic_torch_save(cache, path)
    monitor.event(f"حُفظت ميزات التدريب والتحقق فقط: {path.name}")
    return cache, path
