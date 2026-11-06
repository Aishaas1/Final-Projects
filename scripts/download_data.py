"""Public Kaggle download with verified inventory and actionable, safe errors."""
import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facejpeg.data import inventory
from facejpeg.runtime import Monitor

DATASET = "xhlulu/140k-real-and-fake-faces"


def classify_download_error(exc):
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc)); parts.append(str(exc).lower())
        exc = exc.__cause__ or exc.__context__
    text = " ".join(parts)  # Classify in memory; never save arbitrary exception URLs/headers.
    if "403" in text and ("tunnel" in text or "proxy" in text):
        return "network_policy", "منع وكيل الشبكة الوصول إلى Kaggle (403). هذا ليس دليلًا على غياب مفتاح؛ راجع نطاقات الشبكة أو استخدم ZIP/Drive."
    if "401" in text or "unauthenticated" in text or "authentication" in text:
        return "authentication", "يلزم تسجيل دخول Kaggle/اعتماد صالح بالطريقة الرسمية. لا تكتب المفاتيح في الدفتر؛ استخدم آلية Kaggle الآمنة أو نزّل ZIP من حسابك."
    if "403" in text or "permission" in text:
        return "access_denied", "رفض Kaggle الإذن (403): راجع حسابك وقبول الشروط وصلاحية المجموعة. لا يمكن الجزم أن السر مفقود."
    if "404" in text or "not found" in text:
        return "not_found", "لم يعثر Kaggle على المجموعة/الإصدار أو لم يُتح لك الوصول؛ راجع رابط المصدر."
    if "ssl" in text or "certificate" in text:
        return "tls", "فشل التحقق من شهادة الاتصال؛ أصلح مخزن الثقة ولا تعطل TLS."
    return "download_or_layout", "لم يكتمل التنزيل أو لم نجد صورًا في real وfake. افحص المخرجات والمسار؛ لا يُعد التنزيل ناجحًا قبل فحص الفئتين."


def download(output):
    import kagglehub
    record = Path(output)
    record.parent.mkdir(parents=True, exist_ok=True)
    with Monitor(record.parent / "download_monitor") as monitor:
        monitor.stage("تنزيل Kaggle — قد يشمل الانتظار/المصادقة/فك الملفات")
        try:
            root = Path(kagglehub.dataset_download(DATASET))
            monitor.stage("التحقق من وجود ملفات الفئتين بعد التنزيل")
            _, counts = inventory(root)
        except Exception as exc:
            reason, message = classify_download_error(exc)
            record.write_text(json.dumps({"dataset": DATASET, "status": "failed", "reason": reason,
                "message_ar": message}, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(message + " اختر SOURCE='zip' أو 'drive' في إعدادات الدفتر للمتابعة.") from None
        record.write_text(json.dumps({"dataset": DATASET, "root": str(root),
            "status": "files_found_pending_image_audit", "counts": counts}, indent=2), encoding="utf-8")
        monitor.event(f"ملفات الفئتين موجودة: {counts}؛ فحص سلامة الصور والتكرارات هو المرحلة التالية.")
        print(f"RAW_ROOT = {str(root)!r}")
        return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="data/download.json")
    a = p.parse_args()
    download(a.record)
