"""Predeclared baseline/JPEG training comparison with a test-time model lock."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .augmentation import jpeg_settings
from .common import file_sha256, load_bundle, write_json
from .data import validate_manifest
from .evaluate import QUALITIES, evaluate
from .train import train

MODELS = ("baseline", "jpeg_augmented")


def _atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    write_json(temp, data)
    temp.replace(path)


def _locked_models(prepared, output):
    root, out = Path(prepared), Path(output)
    path = out / "models_locked.json"
    if not path.is_file():
        raise RuntimeError("درّب النموذجين واختر حدود التحقق أولًا؛ لم يُفتح الاختبار.")
    lock = json.loads(path.read_text())
    if lock["manifest_sha256"] != file_sha256(root / "manifest.csv"):
        raise ValueError("Comparison split changed; use the original prepared data")
    checkpoints = {}
    for name in MODELS:
        checkpoint = out / name / "best.pt"
        if not checkpoint.is_file() or file_sha256(checkpoint) != lock["models"][name]["checkpoint_sha256"]:
            raise ValueError("A locked checkpoint changed; do not tune models after opening test results")
        checkpoints[name] = checkpoint
    return checkpoints, lock


def fit_jpeg_comparison(prepared, output, jpeg_qualities=(90, 70, 50, 30), jpeg_probability=.5, **training_options):
    """Fit/calibrate BOTH models. No test prediction is made by this function."""
    qualities, probability = jpeg_settings(jpeg_qualities, jpeg_probability)
    if not qualities or probability <= 0:
        raise ValueError("The JPEG comparison needs at least one quality and a positive probability")
    root, out = Path(prepared), Path(output)
    validate_manifest(root)
    plan = {"schema": 1, "manifest_sha256": file_sha256(root / "manifest.csv"),
            "jpeg_qualities": list(qualities), "jpeg_probability": probability,
            "training_options": training_options, "test_qualities": QUALITIES,
            "validation": "clean original validation, identical for both models",
            "selection": "each best checkpoint and thresholds from validation; fixed across qualities",
            "test_policy": "no test predictions until both checkpoints are calibrated and locked"}
    plan_path = out / "comparison_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Comparison plan changed; choose a new experiment name instead of reusing test results")
    _atomic_json(plan_path, plan)
    if (out / "models_locked.json").exists():
        checkpoints, _ = _locked_models(root, out)
        return checkpoints
    checkpoints = {}
    lock = {"manifest_sha256": plan["manifest_sha256"], "models": {},
            "locked_before_test": True, "validation_images_compressed": False,
            "test_images_used_for_optimization": False}
    for name in MODELS:
        print(f"\n{'A — النموذج الأساسي' if name == 'baseline' else 'B — تدريب JPEG'}", flush=True)
        checkpoint = train(root, out / name, jpeg_qualities=qualities if name == "jpeg_augmented" else (),
                           jpeg_probability=probability, **training_options)
        # load_bundle also verifies calibrated thresholds, preprocessing, and class order.
        model, bundle = load_bundle(checkpoint, "cpu")
        del model
        checkpoints[name] = checkpoint
        lock["models"][name] = {"checkpoint_sha256": file_sha256(checkpoint),
                                "thresholds": bundle["thresholds"], "run_label": bundle["run_label"],
                                "data_kind": bundle["data_kind"]}
    _atomic_json(out / "models_locked.json", lock)
    print("اكتمل تدريب النموذجين وتثبيت حدودهما من التحقق؛ لم نحسب تنبؤات الاختبار بعد.")
    return checkpoints


def evaluate_jpeg_comparison(prepared, output, device="cpu", batch_size=32):
    """Evaluate the frozen pair on the same test images and independent JPEG views."""
    root, out = Path(prepared), Path(output)
    checkpoints, lock = _locked_models(root, out)
    tables = {name: evaluate(root, checkpoints[name], out / name / "evaluation",
                             device=device, batch_size=batch_size, qualities=QUALITIES) for name in MODELS}
    counts = ["fake_detected", "real_falsely_accused", "abstain_total"]
    rates = ["fake_detection_rate_all_fake", "false_accusation_rate_all_real", "abstain_rate_all"]
    records = []
    for index in range(len(QUALITIES)):
        a, b = (tables[name].iloc[index] for name in MODELS)
        if any(a[key] != b[key] for key in ["quality", "n_total", "n_real", "n_fake"]):
            raise ValueError("Comparison requires the same test images and quality order")
        record = {key: a[key] for key in ["quality", "n_total", "n_real", "n_fake"]}
        for name in MODELS:
            row = tables[name].iloc[index]
            for key in counts + rates + ["binary_accuracy_at_0_5_all", "binary_f1_at_0_5", "roc_auc_all"]:
                record[f"{name}_{key}"] = row[key]
        for key in rates:
            record[f"{key}_delta_pp"] = 100 * (b[key] - a[key])
        records.append(record)
    table = pd.DataFrame(records)
    table.to_csv(out / "comparison.csv", index=False)
    write_json(out / "comparison_protocol.json", {**lock, "qualities": QUALITIES,
        "comparison": "B JPEG-trained minus A baseline; deltas in percentage points",
        "not_a_face_benchmark": any(lock["models"][name]["data_kind"] != "research_faces" for name in MODELS)})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(table))
    for ax, rate, title in zip(axes, rates, ["Fake detected / all fake", "Real accused / all real", "Inconclusive / all images"]):
        for name in MODELS:
            ax.plot(x, 100 * table[f"{name}_{rate}"], "o-", label=name)
        ax.set(xticks=x, xticklabels=table.quality, ylabel="Percent", ylim=(-2, 102), title=title)
        ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparison.png", dpi=150); plt.close(fig)
    lines = ["# مقارنة التدريب الأساسي والتدريب باستخدام JPEG", "",
             "A: تدريب أساسي. B: ضغط JPEG عشوائي لصور التدريب فقط؛ مع التقسيم وإعدادات التدريب والبذرة نفسها.",
             "درّبنا النموذجين واخترنا كل أفضل نموذج وحدوده من التحقق الأصلي، ثم ثبّتناهما قبل أي تنبؤ اختبار. حدود كل نموذج ثابتة عبر الجودات، وقد تختلف بين النموذجين.",
             "كل نسخة اختبار تُنشأ مباشرة من الأصل الموحد224×224. جودة100 ليست الأصل غير المعاد ترميزه. هذه مقارنة واحدة ببذرة واحدة، ولا تثبت تفوقًا عامًا أو دلالة إحصائية.", ""]
    if any(lock["models"][name]["data_kind"] != "research_faces" for name in MODELS):
        lines += ["**اختبار برمجيات على أشكال هندسية فقط؛ ليست نتائج كشف وجوه ولا قياسًا لتحسن دقتها.**", ""]
    for index in range(len(QUALITIES)):
        a, b = (tables[name].iloc[index] for name in MODELS)
        lines.append(f"- الجودة {a.quality}: اكتشاف المولّد A={int(a.fake_detected)}/{int(a.n_fake)} وB={int(b.fake_detected)}/{int(b.n_fake)}؛ الاتهام الخاطئ A={int(a.real_falsely_accused)}/{int(a.n_real)} وB={int(b.real_falsely_accused)}/{int(b.n_real)}؛ غير الحاسم A={int(a.abstain_total)}/{int(a.n_total)} وB={int(b.abstain_total)}/{int(b.n_total)}. فرق B−A في الاكتشاف {100*(b.fake_detection_rate_all_fake-a.fake_detection_rate_all_fake):+.1f} نقطة مئوية، وفي الاتهام الخاطئ {100*(b.false_accusation_rate_all_real-a.false_accusation_rate_all_real):+.1f} نقطة.")
    lines += ["", "ارتفاع الاكتشاف مفيد، وارتفاع الاتهام الخاطئ غير مرغوب. قد يزيد غير الحاسم أو ينقص؛ لا نعلن التحسن من رقم منفرد. النسب مقامها جميع صور الفئة أو جميع الاختبار، وليست الحالات المحسومة فقط.",
              "لا نعدّل الإعدادات بناءً على هذا الاختبار. تطوير تجربة لاحقة يحتاج تحققًا مستقلًا، واختبارًا نهائيًا لم يُستخدم لاختيار الإعدادات."]
    (out / "COMPARISON_AR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table
