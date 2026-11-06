import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .common import (CLASSES, decisions, file_sha256, jpeg_copy, load_bundle,
                     read_rgb, tensorize, write_json)

QUALITIES = ["original", 90, 70, 50, 30]


def summary_metrics(labels, scores, pred):
    y, s, d = np.asarray(labels), np.asarray(scores), np.asarray(pred)
    nreal, nfake = int((y == 0).sum()), int((y == 1).sum())
    tp = int(((y == 1) & (d == 1)).sum())
    fp = int(((y == 0) & (d == 1)).sum())
    ar = int(((y == 0) & (d == 2)).sum())
    af = int(((y == 1) & (d == 2)).sum())
    accepted = int((d != 2).sum())
    return {"n_total": len(y), "n_real": nreal, "n_fake": nfake, "fake_detected": tp,
            "fake_detection_rate_all_fake": tp / nfake, "real_falsely_accused": fp,
            "false_accusation_rate_all_real": fp / nreal,
            "abstain_real": ar, "abstain_fake": af, "abstain_total": ar + af,
            "abstain_rate_all": (ar + af) / len(y), "coverage_all": accepted / len(y),
            "selective_accuracy_accepted": float((d == y).sum() / accepted) if accepted else None,
            "precision_predicted_fake": tp / (tp + fp) if tp + fp else None,
            "roc_auc_all": float(roc_auc_score(y, s)),
            "binary_accuracy_at_0_5_all": float(((s >= .5).astype(int) == y).mean())}


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
    lines += ["", f"عند الجودة 30 تغيّر معدل اكتشاف المولّد بمقدار {delta:+.1f} نقطة مئوية، ومعدل الاتهام الخاطئ بمقدار {dfp:+.1f} نقطة مقارنة بالأصل.",
        "هذه مقارنة زوجية على الصور نفسها. لا يلزم أن يكون الأثر رتيبًا مع الجودة، ولا يجوز تعميمه على مولّدات أخرى أو صور الإنترنت. لا يمكن بهذه التجربة وحدها فصل بصمات المولّد عن بصمات مصدر البيانات.",
        "العينة الصغيرة والتكرارات القريبة غير المكتشفة وغياب بيانات الهوية تحدّ الاستنتاج. يُفضّل تكرار التجربة على مصدر خارجي ومولّد غير مستخدم في التدريب."]
    return "\n".join(lines)


def evaluate(prepared, checkpoint, output, batch_size=32, device=None):
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
    test = pd.read_csv(root / "manifest.csv").query("split == 'test'").reset_index(drop=True)
    all_predictions, summaries, matrices = [], [], {}
    for quality in QUALITIES:
        scores = []
        for start in range(0, len(test), batch_size):
            ims = [read_rgb(root / p) for p in test.path.iloc[start:start+batch_size]]
            if quality != "original":
                ims = [jpeg_copy(im, quality) for im in ims]
            x = torch.stack([tensorize(im, bundle["preprocess"]) for im in ims]).to(device)
            with torch.inference_mode():
                scores.extend(model(x).flatten().sigmoid().cpu().tolist())
        pred = decisions(scores, bundle["thresholds"])
        records = pd.DataFrame({"path": test.path, "label": test.label,
                                "quality": str(quality), "score_fake": scores, "decision": pred})
        all_predictions.append(records)
        summaries.append({"quality": str(quality), **summary_metrics(test.label, scores, pred)})
        cm = np.zeros((2, 3), dtype=int)
        for y, d in zip(test.label, pred):
            cm[int(y), int(d)] += 1
        matrices[str(quality)] = cm.tolist()
    predictions, table = pd.concat(all_predictions, ignore_index=True), pd.DataFrame(summaries)
    predictions.to_csv(out / "predictions.csv", index=False)
    table.to_csv(out / "jpeg_metrics.csv", index=False)
    write_json(out / "confusion_matrices.json", {"rows": CLASSES, "columns": CLASSES + ["inconclusive"], "matrices": matrices})
    write_json(out / "evaluation_protocol.json", {"thresholds": bundle["thresholds"],
        "checkpoint_sha256": file_sha256(checkpoint), "manifest_sha256": bundle["manifest_sha256"],
        "qualities": QUALITIES, "run_label": bundle["run_label"], "test_count": len(test)})
    x = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, label in [("fake_detection_rate_all_fake", "Fake detected / all fake"),
                       ("false_accusation_rate_all_real", "Real accused / all real"),
                       ("abstain_rate_all", "Inconclusive / all images")]:
        ax.plot(x, table[col] * 100, "o-", label=label)
    ax.set(xticks=x, xticklabels=table.quality, xlabel="JPEG quality (independent re-encodes)",
           ylabel="Percent", ylim=(-2, 102), title=bundle["run_label"])
    ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out / "jpeg_effect.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, quality in zip(axes, QUALITIES):
        cm = np.asarray(matrices[str(quality)])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(test.label.value_counts()))
        ax.set(xticks=[0, 1, 2], xticklabels=["Real", "Fake", "Unsure"],
               yticks=[0, 1], yticklabels=["Real", "Fake"], title=str(quality), xlabel="Predicted")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
    axes[0].set_ylabel("True class")
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
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate(a.prepared, a.checkpoint, a.output)
