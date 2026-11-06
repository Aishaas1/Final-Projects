import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

CLASSES = ["real", "fake"]
LABELS_AR = {0: "يرجّح أنها صورة فوتوغرافية", 1: "يرجّح أنها مولَّدة بالذكاء الاصطناعي", 2: "النتيجة غير حاسمة"}
PREPROCESS = {"size": 224, "resize": "center_square_bicubic", "color": "RGB",
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
              "exif_transpose": True, "jpeg_subsampling": 2}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_rgb(path):
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")


def canonical(im, config=None):
    cfg = config or PREPROCESS
    return ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"),
                        (cfg["size"], cfg["size"]), method=Image.Resampling.BICUBIC)


def jpeg_copy(im, quality):
    """Always call on original RGB pixels, never on another JPEG copy."""
    quality = int(quality)
    if not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be in [1,100]")
    stream = io.BytesIO()
    im.convert("RGB").save(stream, format="JPEG", quality=quality, subsampling=2, optimize=False)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert("RGB").copy()


def tensorize(im, config=None):
    cfg = config or PREPROCESS
    return transforms.Normalize(cfg["mean"], cfg["std"])(transforms.ToTensor()(canonical(im, cfg)))


def build_model(pretrained=False):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    # Torchvision verifies the official weight filename's SHA256 prefix.
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 1)
    return model


def decisions(scores, thresholds):
    s = np.asarray(scores)
    return np.where(s <= thresholds["low"], 0, np.where(s >= thresholds["high"], 1, 2))


def atomic_torch_save(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temp)
    temp.replace(path)


def load_bundle(path, device="cpu"):
    # Only tensor/primitive checkpoints; do not load arbitrary pickled objects.
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != 1 or bundle.get("classes") != CLASSES:
        raise ValueError("Unsupported checkpoint schema/class order")
    if bundle.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Unsupported architecture")
    if bundle.get("thresholds") is None:
        raise ValueError("Select validation thresholds before inference")
    low, high = bundle["thresholds"].get("low"), bundle["thresholds"].get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)) or not 0 <= low < high <= 1:
        raise ValueError("Invalid checkpoint decision thresholds")
    cfg = bundle.get("preprocess", {})
    if cfg.get("resize") != "center_square_bicubic" or cfg.get("size") != 224 or cfg.get("color") != "RGB":
        raise ValueError("Unsupported checkpoint preprocessing")
    mean, std = cfg.get("mean", []), cfg.get("std", [])
    if (not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)) or len(mean) != 3 or len(std) != 3
            or not all(isinstance(v, (int, float)) and np.isfinite(v) for v in [*mean, *std])
            or any(s <= 0 for s in std)):
        raise ValueError("Invalid normalization parameters")
    model = build_model(False)
    model.load_state_dict(bundle["model_state"])
    return model.to(device).eval(), bundle


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
