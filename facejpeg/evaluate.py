import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tqdm.auto import tqdm

from .common import (CLASSES, decisions, file_sha256, jpeg_copy, load_bundle,
                     read_rgb, tensorize, write_json)
from .data import validate_manifest
from .runtime import Monitor

QUALITIES = ["original", 100, 90, 70, 50, 30, 10]


def binary_metrics(labels, scores):
    y, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not len(y) or len(y) != len(scores) or not np.isfinite(scores).all() or not set(y).issubset({0, 1}):
        raise ValueError("Metrics require nonempty finite aligned binary data")
    pred = (scores >= .5).astype(int)
    return {"binary_accuracy_at_0_5_all": float(accuracy_score(y, pred)),
            "binary_precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
            "binary_recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
            "binary_f1_at_0_5": float(f1_score(y, pred, zero_division=0)),
            "binary_macro_f1_at_0_5": float(f1_score(y, pred, average="macro", zero_division=0)),
            "roc_auc_all": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None}


def summary_metrics(labels, scores, pred):
    y, s, d = np.asarray(labels), np.asarray(scores), np.asarray(pred)
    nreal, nfake = int((y == 0).sum()), int((y == 1).sum())
    tp = int(((y == 1) & (d == 1)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    ar = int(((y == 0) & (d == 2)).sum())
    af = int(((y == 1) & (d == 2)).sum())
    accepted = int((d != 2).sum())
    if not nreal or not nfake or not set(d).issubset({0, 1, 2}):
        raise ValueError("JPEG summary requires both classes and three-way decisions")
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / nfake
    return {"n_total": len(y), "n_real": nreal, "n_fake": nfake, "fake_detected": tp,
            "fake_detection_rate_all_fake": tp / nfake, "real_falsely_accused": fp,
            "false_accusation_rate_all_real": fp / nreal,
            "abstain_real": ar, "abstain_fake": af, "abstain_total": ar + af,
            "abstain_rate_all": (ar + af) / len(y), "coverage_all": accepted / len(y),
            "selective_accuracy_accepted": float((d == y).sum() / accepted) if accepted else None,
            "precision_predicted_fake": precision,
            "fake_f1_with_abstention": 2 * precision * recall / (precision + recall) if precision is not None and precision+recall else 0.,
            **binary_metrics(y, s)}


def report_ar(table, run_label):
    base = table.iloc[0]
    lines = [f"# تقرير تجربة JPEG — {run_label}", "",
        "الدرجة ترجيح النموذج للفئة المولّدة، وليست إثباتًا للأصالة أو احتمالًا معايرًا.",
        "خط الأساس original هو الأصل المفكوك بعد توحيد RGB والقص المركزي إلى 224×224، دون إعادة JPEG إضافية. كل جودة أُنشئت مستقلة منه، وليست سلسلة ضغط. قد يكون الأصل نفسه JPEG سابقًا.", "",
        "نسبة اكتشاف المولّد = المولّد المحكوم عليه fake ÷ جميع المولّدات؛ الاتهام الخاطئ = الحقيقي المحكوم عليه fake ÷ جميع الصور الحقيقية. غير الحاسم يُعدّ ضمن المقام ولا يُحتسب اكتشافًا صحيحًا. الدقة الانتقائية مقامها الحالات المحسومة فقط؛ AUC تستخدم جميع الدرجات.", ""]
    if run_label == "SMOKE_ONLY_NOT_A_FACE_DETECTOR":
        lines += ["**هذه بيانات هندسية اصطناعية لاختبار البرمجيات فقط. لا تمثل صور أشخاص أو قدرة كشف التوليد. لا تُقدّم أرقامها كنتيجة علمية للمشروع.**", ""]
    for _, r in table.iterrows():
        lines.append(f"- الجودة {r.quality}: اكتشف {int(r.fake_detected)}/{int(r.n_fake)} مولّدة ({100*r.fake_detection_rate_all_fake:.1f}%)؛ اتّهم خطأً {int(r.real_falsely_accused)}/{int(r.n_real)} فوتوغرافية ({100*r.false_accusation_rate_all_real:.1f}%)؛ غير حاسم {int(r.abstain_total)}/{int(r.n_total)} ({100*r.abstain_rate_all:.1f}%).")
    last = table.iloc[-1]
    delta = 100 * (last.fake_detection_rate_all_fake - base.fake_detection_rate_all_fake)
    dfp = 100 * (last.false_accusation_rate_all_real - base.false_accusation_rate_all_real)
    lines += ["", f"عند الجودة {last.quality} تغيّر معدل اكتشاف المولّد بمقدار {delta:+.1f} نقطة مئوية، ومعدل الاتهام الخاطئ بمقدار {dfp:+.1f} نقطة مقارنة بالأصل.",
        f"التصنيف الثنائي المساعد عند 0.5 على الأصل: Accuracy={base.binary_accuracy_at_0_5_all:.4f}، Precision={base.binary_precision_at_0_5:.4f}، Recall={base.binary_recall_at_0_5:.4f}، F1={base.binary_f1_at_0_5:.4f}، AUC={base.roc_auc_all:.4f}. هذه المقاييس لا تمثل قرار الواجهة الثلاثي؛ عمود fake_f1_with_abstention يحسب غير الحاسم المولّد ضمن ما لم يُكتشف. عند غياب توقعات موجبة نُعرّف Precision وF1 الثنائيين بصفر (zero_division=0).",
        "هذه مقارنة زوجية على الصور نفسها. لا يلزم أن يكون الأثر رتيبًا مع الجودة، ولا يجوز تعميمه على مولّدات أخرى أو صور الإنترنت. لا يمكن بهذه التجربة وحدها فصل بصمات المولّد عن بصمات مصدر البيانات.",
        "العينة الصغيرة والتكرارات القريبة غير المكتشفة وغياب بيانات الهوية تحدّ الاستنتاج. يُفضّل تكرار التجربة على مصدر خارجي ومولّد غير مستخدم في التدريب."]
    return "\n".join(lines)


def evaluate(prepared, checkpoint, output, batch_size=32, device=None, qualities=None):
    with Monitor(output) as monitor:
        return _evaluate(prepared, checkpoint, output, batch_size, device, QUALITIES if qualities is None else qualities, monitor)


def _evaluate(prepared, checkpoint, output, batch_size, device, qualities, monitor):
    if not qualities or qualities[0] != "original" or len(set(map(str, qualities))) != len(qualities):
        raise ValueError("Qualities must start with original and contain no duplicates")
    for q in qualities[1:]:
        if isinstance(q, bool) or not isinstance(q, int) or not 1 <= q <= 100:
            raise ValueError("JPEG qualities must be integers in [1,100]")
    root, out = Path(prepared), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    # Clear only our mutually exclusive generated examples when reevaluating this run.
    for name in ["changed_examples.png", "NO_CHANGED_EXAMPLES.txt"]:
        (out / name).unlink(missing_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, bundle = load_bundle(checkpoint, device)
    if file_sha256(root / "manifest.csv") != bundle["manifest_sha256"]:
        raise ValueError("Test manifest differs from checkpoint's split")
    test = validate_manifest(root, verify_files=True).query("split == 'test'").reset_index(drop=True)
    all_predictions, summaries, matrices = [], [], {}
    roc_records = []
    for quality in qualities:
        monitor.stage(f"تقييم الاختبار JPEG={quality}", len(test))
        scores = []
        for start in tqdm(range(0, len(test), batch_size), desc=f"Test Q={quality}", unit="batch"):
            ims = [read_rgb(root / p) for p in test.path.iloc[start:start+batch_size]]
            if quality != "original":
                ims = [jpeg_copy(im, quality) for im in ims]
            x = torch.stack([tensorize(im, bundle["preprocess"]) for im in ims]).to(device)
            with torch.inference_mode():
                scores.extend(model(x).flatten().sigmoid().cpu().tolist())
            monitor.advance(len(ims), f"test quality={quality} images={len(scores)}/{len(test)}")
        pred = decisions(scores, bundle["thresholds"])
        records = pd.DataFrame({"path": test.path, "label": test.label,
                                "quality": str(quality), "score_fake": scores, "decision": pred})
        all_predictions.append(records)
        summaries.append({"quality": str(quality), **summary_metrics(test.label, scores, pred)})
        cm = np.zeros((2, 3), dtype=int)
        for y, d in zip(test.label, pred):
            cm[int(y), int(d)] += 1
        matrices[str(quality)] = cm.tolist()
        fpr, tpr, thresholds = roc_curve(test.label, scores)
        roc_records.extend({"quality": str(quality), "fpr": float(a), "tpr": float(b), "threshold": float(c)}
                           for a, b, c in zip(fpr, tpr, thresholds))
    predictions, table = pd.concat(all_predictions, ignore_index=True), pd.DataFrame(summaries)
    predictions.to_csv(out / "predictions.csv", index=False)
    table.to_csv(out / "jpeg_metrics.csv", index=False)
    write_json(out / "confusion_matrices.json", {"rows": CLASSES, "columns": CLASSES + ["inconclusive"], "matrices": matrices})
    write_json(out / "evaluation_protocol.json", {"thresholds": bundle["thresholds"],
        "checkpoint_sha256": file_sha256(checkpoint), "manifest_sha256": bundle["manifest_sha256"],
        "qualities": qualities, "run_label": bundle["run_label"], "test_count": len(test),
        "decision_threshold_source": "validation only; unchanged across JPEG qualities",
        "binary_metrics_policy": "threshold 0.5; zero_division=0; separate from three-way UI"})
    monitor.stage("حفظ الجداول والرسوم والتقرير")
    roc_table = pd.DataFrame(roc_records)
    roc_table.to_csv(out / "roc_points.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    for _, r in table.iterrows():
        part = roc_table[roc_table.quality == r.quality]
        ax.plot(part.fpr, part.tpr, label=f"{r.quality}: AUC={r.roc_auc_all:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set(xlabel="False positive rate / all real", ylabel="True positive rate / all fake",
           title=bundle["run_label"])
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=150); plt.close(fig)
    delta = table.copy()
    for column in ["fake_detection_rate_all_fake", "false_accusation_rate_all_real", "abstain_rate_all", "binary_f1_at_0_5"]:
        delta[column + "_change_pp"] = 100 * (table[column] - table.iloc[0][column])
    delta["roc_auc_change"] = table.roc_auc_all - table.iloc[0].roc_auc_all
    delta.to_csv(out / "jpeg_changes.csv", index=False)
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in [("fake_detection_rate_all_fake", "Fake detected / all fake"),
                       ("false_accusation_rate_all_real", "Real accused / all real"),
                       ("abstain_rate_all", "Inconclusive / all images")]:
        ax.plot(x, table[col] * 100, "o-", label=label)
    ax.set(xticks=x, xticklabels=table.quality, xlabel="JPEG quality (independent re-encodes)",
           ylabel="Percent", ylim=(-2, 102), title=bundle["run_label"])
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "jpeg_effect.png", dpi=160); plt.close(fig)
    cols = min(4, len(qualities))
    rows = (len(qualities) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.4), squeeze=False)
    for ax, quality in zip(axes.flat, qualities):
        cm = np.asarray(matrices[str(quality)])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(test.label.value_counts()))
        ax.set(xticks=[0, 1, 2], xticklabels=["Real", "Fake", "Unsure"],
               yticks=[0, 1], yticklabels=["Real", "Fake"], title=str(quality), xlabel="Predicted")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
    for ax in list(axes.flat)[len(qualities):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("True class")
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=160); plt.close(fig)
    original = predictions[predictions.quality == "original"].set_index("path")
    changed = predictions[predictions.quality != "original"].copy()
    changed["original_decision"] = changed.path.map(original.decision)
    changed["original_score"] = changed.path.map(original.score_fake)
    changed = changed[changed.decision != changed.original_decision]
    changed.to_csv(out / "changed_decisions.csv", index=False)
    if len(changed):
        fig, axes = plt.subplots(min(6, len(changed)), 2, figsize=(8, 3.5 * min(6, len(changed))), squeeze=False)
        for axes_row, (_, r) in zip(axes, changed.head(6).iterrows()):
            im = read_rgb(root / r.path)
            for ax, pic, title in zip(axes_row, [im, jpeg_copy(im, int(r.quality))],
                [f"Original: d={r.original_decision}, s={r.original_score:.3f}", f"Q={r.quality}: d={r.decision}, s={r.score_fake:.3f}; true={r.label}"]):
                ax.imshow(pic); ax.set_title(title); ax.axis("off")
        fig.tight_layout(); fig.savefig(out / "changed_examples.png", dpi=130); plt.close(fig)
    else:
        (out / "NO_CHANGED_EXAMPLES.txt").write_text("لم يتغير الحكم الثلاثي لأي صورة في هذه التجربة؛ راجع تغير الدرجات في predictions.csv. لا نختلق أمثلة.\n", encoding="utf-8")
    (out / "REPORT_AR.md").write_text(report_ar(table, bundle["run_label"]), encoding="utf-8")
    monitor.event(f"اكتمل تقييم {len(test)} صورة × {len(qualities)} حالات؛ لم تُعدّل حدود التحقق")
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.prepared, a.checkpoint, a.output)
