"""Audit before splitting; deduplicate/group BEFORE any JPEG experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from .common import CLASSES, canonical, file_sha256, read_rgb, write_json
from .runtime import Monitor


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"مجلد البيانات غير موجود: {root}")
    paths, counts = [], {"real": 0, "fake": 0, "ambiguous": 0, "contains_procedural_fixture": False}
    for p in root.rglob("*"):
        if p.is_file() and p.name in {"_fixture_metadata.json", "audit.json"}:
            if not p.resolve().is_relative_to(root):
                raise ValueError("Dataset metadata symlink escapes data root")
            if p.name == "_fixture_metadata.json":
                counts["contains_procedural_fixture"] = True
            else:
                try:
                    if json.loads(p.read_text(encoding="utf-8")).get("data_kind") == "procedural_fixture":
                        counts["contains_procedural_fixture"] = True
                except (ValueError, AttributeError):
                    pass
        if not p.is_file() or p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if not p.resolve().is_relative_to(root):
            raise ValueError("Image symlink escapes data root")
        paths.append(p)
        labels = {part.lower() for part in p.relative_to(root).parts[:-1]} & set(CLASSES)
        counts[next(iter(labels)) if len(labels) == 1 else "ambiguous"] += 1
    if not counts["real"] or not counts["fake"]:
        raise ValueError(f"يلزم مجلدا real وfake وصور في كل فئة. الموجود: {counts}")
    return sorted(paths), counts


def validate_manifest(root):
    root = Path(root).resolve()
    frame = pd.read_csv(root / "manifest.csv", keep_default_na=False,
                        dtype={k: str for k in ["path", "split", "group", "pixel_sha256", "file_sha256", "identity", "source_id", "phash"]})
    required = {"path", "label", "split", "group", "pixel_sha256"}
    if not required.issubset(frame.columns) or frame.empty or frame.path.duplicated().any():
        raise ValueError("Invalid or empty manifest; repeat preparation in a new directory")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Expected train/val/test only")
    for split, part in frame.groupby("split"):
        if set(part.label) != {0, 1}:
            raise ValueError(f"Missing class in {split}")
    for key in ["group", "pixel_sha256", "file_sha256", "identity", "source_id", "phash"]:
        if key in frame:
            nonempty = frame[frame[key].astype(str) != ""]
            if nonempty.groupby(key).split.nunique().max() > 1:
                raise ValueError(f"Data leakage: {key} spans multiple splits")
    for rel in frame.path:
        p = (root / rel).resolve()
        if Path(rel).is_absolute() or not p.is_relative_to(root) or not p.is_file():
            raise ValueError(f"Missing/unsafe prepared image: {rel}")
    return frame


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        node = (value, index, {})
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            d = (value ^ cur[0]).bit_count()
            if d not in cur[2]:
                cur[2][d] = node
                return
            cur = cur[2][d]

    def query(self, value, radius):
        stack, result = [self.root] if self.root else [], []
        while stack:
            v, idx, children = stack.pop()
            d = (value ^ v).bit_count()
            if d <= radius:
                result.append(idx)
            stack.extend(child for edge, child in children.items() if d-radius <= edge <= d+radius)
        return result


def safe_extract(zip_path, destination):
    import zipfile
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        if sum(i.file_size for i in z.infolist()) > 40 * 1024**3:
            raise ValueError("ZIP expands beyond 40 GiB; extract manually after inspection")
        for info in z.infolist():
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP path/symlink")
            if target.exists() and not info.is_dir():
                raise FileExistsError(f"Refusing overwrite: {target}")
        z.extractall(root)
    return root


def prepare(root, output, metadata=None, seed=42, max_per_class=None, phash_radius=4,
            data_kind="research_faces"):
    with Monitor(output) as monitor:
        return _prepare(root, output, metadata, seed, max_per_class, phash_radius, data_kind, monitor)


def _prepare(root, output, metadata, seed, max_per_class, phash_radius, data_kind, monitor):
    root, output = Path(root).resolve(), Path(output).resolve()
    if (root / "_fixture_metadata.json").exists():
        data_kind = "procedural_fixture"  # Keep our generated test data visibly marked.
    if output.is_relative_to(root):
        raise ValueError("Prepared output must be outside the raw-data directory")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.csv").exists():
        raise FileExistsError("Use the existing manifest or choose a new output directory")
    meta = {}
    if metadata:
        md = pd.read_csv(metadata, dtype=str, keep_default_na=False)
        if "path" not in md or md.path.duplicated().any():
            raise ValueError("metadata.csv requires unique relative 'path' entries")
        meta = md.set_index("path").to_dict("index")
    monitor.stage("حصر ملفات البيانات")
    paths, counts = inventory(root)
    if counts["contains_procedural_fixture"]:
        data_kind = "procedural_fixture"
    write_json(output / "inventory.json", counts)
    monitor.event(f"عُثر على ملفات الفئتين (ليست فحص صلاحية بعد): {counts}")
    monitor.stage("فحص الصور والتلف والبصمات", len(paths))
    rows, bad = [], []
    for path in tqdm(paths, desc="Audit images", unit="image"):
        rel = path.relative_to(root).as_posix()
        labels = {p.lower() for p in Path(rel).parts[:-1]} & set(CLASSES)
        if len(labels) != 1:
            bad.append({"path": rel, "reason": "missing/ambiguous real/fake folder"})
            monitor.advance(message=f"rejected unlabelled {path.name}")
            continue
        try:
            with Image.open(path) as im:
                fmt, width, height = im.format, im.width, im.height
                im.verify()
            rgb = read_rgb(path)
            # Content digest catches identical decoded pixels; pHash catches many resizes/JPEG variants.
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            phash = int(str(imagehash.phash(canonical(rgb))), 16)
            row = {"source_path": rel, "label": CLASSES.index(next(iter(labels))),
                   "format": fmt, "width": width, "height": height,
                   "pixel_sha256": digest, "file_sha256": file_sha256(path), "phash": f"{phash:016x}",
                   "identity": meta.get(rel, {}).get("identity", ""),
                   "source_id": meta.get(rel, {}).get("source_id", "")}
            rows.append(row)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            bad.append({"path": rel, "reason": str(exc)})
        monitor.advance(message=f"audited {path.name}")
    write_json(output / "rejected.json", bad)
    if not rows:
        raise ValueError("No readable images in real/ and fake/ folders")
    df = pd.DataFrame(rows)
    df.to_csv(output / "raw_audit.csv", index=False)
    summary = df.groupby(["label", "format"]).agg(count=("width", "size"),
        width_min=("width", "min"), width_median=("width", "median"), width_max=("width", "max"),
        height_min=("height", "min"), height_median=("height", "median"), height_max=("height", "max"))
    summary.to_csv(output / "source_format_resolution.csv")
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    tree, keys, hashes = BKTree(), {}, {}
    monitor.stage("تجميع النسخ والهوية قبل التقسيم", len(rows))
    for i, r in enumerate(tqdm(rows, desc="Group duplicates", unit="image")):
        h = int(r["phash"], 16)
        for j in tree.query(h, phash_radius):
            union(i, j)
        if h not in hashes:
            tree.add(h, i)
            hashes[h] = i
        for field in ["pixel_sha256", "identity", "source_id"]:
            if r[field]:
                key = (field, r[field])
                if key in keys:
                    union(i, keys[key])
                keys[key] = i
        monitor.advance(message=f"grouped image {i+1}")
    df["group"] = [str(find(i)) for i in range(len(rows))]
    conflicts = df.groupby("group").label.nunique()
    conflict_groups = set(conflicts[conflicts > 1].index)
    df[df.group.isin(conflict_groups)].to_csv(output / "label_conflicts.csv", index=False)
    clean = df[~df.group.isin(conflict_groups)].drop_duplicates("pixel_sha256").copy()
    # Class-stratified GROUP allocation; same identity/source/near-duplicate never crosses splits.
    rng = np.random.default_rng(seed)
    assignment = {}
    for label in [0, 1]:
        groups = sorted(clean.loc[clean.label == label, "group"].unique())
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f"Class {CLASSES[label]} needs at least 3 independent groups")
        nt = max(1, round(len(groups) * .15))
        nv = max(1, round(len(groups) * .15))
        for j, group in enumerate(groups):
            assignment[group] = "test" if j < nt else "val" if j < nt + nv else "train"
    clean["split"] = clean.group.map(assignment)
    if max_per_class:
        if max_per_class < 10:
            raise ValueError("max_per_class must be >=10")
        # Only subsample AFTER auditing/group allocation. Preserve class in every split.
        sampled = []
        for (_, split), part in clean.groupby(["label", "split"]):
            cap = max(1, round(max_per_class * {"train": .7, "val": .15, "test": .15}[split]))
            sampled.append(part.sample(min(len(part), cap), random_state=seed))
        clean = pd.concat(sampled).sort_index().copy()
    clean["path"] = ""
    monitor.stage("حفظ المعالجة المشتركة بعد التقسيم", len(clean))
    for idx, row in tqdm(clean.iterrows(), total=len(clean), desc="Normalize split images", unit="image"):
        # Identical processing for both classes; normalized originals stored losslessly AFTER split.
        target = output / "images" / row.split / CLASSES[row.label] / f"{idx:07d}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        canonical(read_rgb(root / row.source_path)).save(target, format="PNG")
        clean.loc[idx, "path"] = target.relative_to(output).as_posix()
        monitor.advance(message=f"saved {target.name}")
    report = {"raw_root": str(root), "readable": len(df), "rejected": len(bad),
              "exact_duplicates_removed": int(len(df[~df.group.isin(conflict_groups)]) - len(df[~df.group.isin(conflict_groups)].drop_duplicates("pixel_sha256"))),
              "conflict_images_excluded": int(df.group.isin(conflict_groups).sum()),
              "kept": len(clean), "seed": seed, "phash_radius": phash_radius,
              "identity_rows": int((df.identity != "").sum()),
              "identity_limitation": "No identity guarantee for rows without identity metadata. pHash is heuristic, not face recognition.",
              "data_kind": data_kind,
              "normalization": "RGB center-square bicubic 224x224 lossless PNG; JPEG only AFTER split",
              "class_counts": {CLASSES[int(k)]: int(v) for k, v in clean.label.value_counts().items()},
              "counts": {str(k): int(v) for k, v in clean.groupby(["split", "label"]).size().items()}}
    write_json(output / "audit.json", report)
    # Manifest is the completion marker; do not write it before images AND audit finish.
    temp = output / "manifest.csv.tmp"
    clean.to_csv(temp, index=False)
    temp.replace(output / "manifest.csv")
    validate_manifest(output)
    monitor.event(f"اكتمل تجهيز {len(clean)} صورة؛ التقسيم والتوازن والتداخل اجتازت الفحص")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return clean


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--metadata")
    p.add_argument("--max-per-class", type=int)
    args = p.parse_args()
    prepare(args.root, args.output, args.metadata, max_per_class=args.max_per_class)
