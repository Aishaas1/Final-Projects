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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.bundle = load_bundle(checkpoint, self.device)

    @torch.inference_mode()
    def score(self, im):
        x = tensorize(im, self.bundle["preprocess"]).unsqueeze(0).to(self.device)
        return float(self.model(x).flatten().sigmoid()[0].cpu())

    def compare(self, path, quality=70):
        raw = validated_image(path)
        original = canonical(raw, self.bundle["preprocess"])
        compressed = jpeg_copy(original, quality)
        scores = [self.score(im) for im in [original, compressed]]
        pred = decisions(scores, self.bundle["thresholds"])
        texts = [f"{LABELS_AR[int(d)]}\n\nدرجة ترجيح التوليد: {s:.4f}" for s, d in zip(scores, pred)]
        low, high = self.bundle["thresholds"]["low"], self.bundle["thresholds"]["high"]
        note = f"الحدود المحفوظة: فوتوغرافية ≤ {low:.3f}، مولّدة ≥ {high:.3f}، وبينهما غير حاسمة. الدرجة ليست إثباتًا لأصالة الصورة."
        if self.bundle["run_label"] == "SMOKE_ONLY_NOT_A_FACE_DETECTOR":
            note = "⚠️ أوزان اختبار برمجي على أشكال اصطناعية فقط؛ هذه الأحكام لا تقيس كشف الوجوه المولّدة. درّب على البيانات الحقيقية قبل تفسيرها.\n\n" + note
        elif self.bundle["quick"]:
            note = "⚠️ نموذج تجربة سريعة على عينة صغيرة؛ نتائجه أولية.\n\n" + note
        return original, compressed, texts[0], texts[1], note
