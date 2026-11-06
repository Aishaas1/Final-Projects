"""Procedural images for software smoke testing, NOT real/generated face research data."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def make_demo(root, count=20):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose an empty fixture directory")
    rng = np.random.default_rng(2026)
    for label in ["real", "fake"]:
        folder = root / label
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            arr = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            im = Image.fromarray(arr)
            draw = ImageDraw.Draw(im)
            color = tuple(int(c) for c in rng.integers(30, 230, 3))
            x, y = int(rng.integers(15, 70)), int(rng.integers(15, 70))
            if label == "real":
                draw.ellipse((x, y, x + 120, y + 120), fill=color)
            else:
                draw.rectangle((x, y, x + 120, y + 120), fill=color)
            im.save(folder / f"fixture_{i:03d}.png")
    (root / "_fixture_metadata.json").write_text(json.dumps({"data_kind": "procedural_fixture",
        "warning": "Software test shapes, not human face detection data"}))
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--count", type=int, default=20)
    a = p.parse_args()
    make_demo(a.root, a.count)
