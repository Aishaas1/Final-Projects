from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .common import LABELS_AR, canonical, decisions, jpeg_copy, load_bundle, read_rgb, tensorize


def face_count(im):
    # Shipped with opencv; no unverified model download. A fallible frontal-face gate, not identity recognition.
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if detector.empty():
        raise RuntimeError("تعذّر تحميل كاشف الوجه من OpenCV.")
    preview = im.copy()
    preview.thumbnail((1024, 1024))
    gray = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return len(boxes)


def validated_image(path):
    if not path:
        raise ValueError("ارفع صورة JPG أو PNG أولًا.")
    p = Path(path)
    if not p.is_file():
        raise ValueError("لم أعثر على ملف الصورة؛ ارفعه مجددًا.")
    if p.stat().st_size > 15 * 1024**2:
        raise ValueError("الملف أكبر من 15 ميغابايت؛ ارفع نسخة أصغر.")
    try:
        with Image.open(p) as im:
            if im.format not in {"JPEG", "PNG"}:
                raise ValueError("الصيغ المدعومة JPG وPNG فقط.")
            if im.width * im.height > 25_000_000:
                raise ValueError("دقة الصورة كبيرة جدًا؛ الحد 25 مليون بكسل.")
            im.verify()
        rgb = read_rgb(p)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("تعذّر قراءة الصورة. الملف تالف أو ليس صورة JPG/PNG سليمة.") from exc
    nfaces = face_count(rgb)
    if nfaces == 0:
        raise ValueError("لم أجد وجهًا أماميًا واضحًا. استخدم صورة أقرب بإضاءة جيدة ووجه واحد في المنتصف؛ قد يُخطئ كاشف الوجه.")
    if nfaces > 1:
        raise ValueError("عُثر على عدة وجوه؛ ارفع صورة تحتوي على وجه واحد فقط.")
    return rgb


class Predictor:
    def __init__(self, checkpoint, device=None):
        if not checkpoint or not Path(checkpoint).is_file():
            raise ValueError("لا توجد أوزان محفوظة. درّب النموذج أولًا ثم اختر best.pt من مجلد التجربة.")
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.model, self.bundle = load_bundle(checkpoint, "cpu")
        cpu_profile = self.bundle["quick"] or self.bundle.get("training_config", {}).get("cache_features", False)
        self.device = device or ("cuda" if torch.cuda.is_available() and not cpu_profile else "cpu")
        self.model.to(self.device)

    @property
    def face_classification_enabled(self):
        return (self.bundle.get("data_kind") == "research_faces"
                and self.bundle.get("run_label") != "SMOKE_ONLY_NOT_A_FACE_DETECTOR")

    @torch.inference_mode()
    def score(self, im):
        x = tensorize(im, self.bundle["preprocess"]).unsqueeze(0).to(self.device)
        return float(self.model(x).flatten().sigmoid()[0].cpu())

    def compare(self, path, quality=70):
        return self.compare_details(path, quality)[:5]

    def compare_details(self, path, quality=70):
        raw = validated_image(path)
        original = canonical(raw, self.bundle["preprocess"])
        compressed = jpeg_copy(original, quality)
        if not self.face_classification_enabled:
            result = "كشف الأصالة غير متاح — النموذج غير مدرّب على صور الوجوه."
            note = ("⚠️ اختبار برمجي: هذه الأوزان لا تدعم الحكم على أصالة الوجه، لذلك لم نحسب درجة Real أو AI-Generated. "
                    "المتاح هنا معاينة الصورة وضغط JPEG فقط. لتفعيل الكشف اختر MODE='REAL'، ووفّر بيانات وجوه حقيقية ومولّدة، "
                    "ودرّب النموذج ثم حمّل best.pt من تلك التجربة. تغيير اسم الوضع وحده لا يحوّل الأوزان القديمة إلى كاشف وجوه.")
            return original, compressed, result, result, note, None, None
        scores = [self.score(im) for im in [original, compressed]]
        pred = decisions(scores, self.bundle["thresholds"])
        texts = [f"{LABELS_AR[int(d)]}\n\nدرجة ترجيح التوليد: {s:.4f}\n\nترجيح الفوتوغرافية: {1-s:.4f}" for s, d in zip(scores, pred)]
        low, high = self.bundle["thresholds"]["low"], self.bundle["thresholds"]["high"]
        note = f"الحدود المحفوظة: فوتوغرافية ≤ {low:.3f}، مولّدة ≥ {high:.3f}، وبينهما غير حاسمة. الدرجة ليست إثباتًا لأصالة الصورة."
        if self.bundle["quick"]:
            note = "⚠️ نموذج تجربة سريعة على عينة صغيرة؛ نتائجه أولية.\n\n" + note
        elif self.bundle["run_label"] == "REAL_CPU_FROZEN_BACKBONE":
            note = "نموذج CPU برأس مدرّب وميزات مجمّدة؛ راجع نتائج الاختبار المستقل وحدود التعميم.\n\n" + note
        distributions = [{"Real — فوتوغرافية": 1-s, "AI-Generated — مولّدة": s} for s in scores]
        return original, compressed, texts[0], texts[1], note, *distributions
