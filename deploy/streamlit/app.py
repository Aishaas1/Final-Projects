"""Free Community Cloud frontend sharing the project's inference implementation."""
import os
from pathlib import Path
import pickle
import sys
import tempfile

import streamlit as st
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facejpeg.common import file_sha256
from facejpeg.infer import Predictor

st.set_page_config(page_title="FaceJPEG — الوجوه وضغط JPEG", page_icon="🖼️", layout="wide")
st.markdown("<style>.stApp {direction:rtl; text-align:right;}</style>", unsafe_allow_html=True)
st.title("كشف صور الوجوه المولّدة ودراسة أثر JPEG")
st.caption("ارفع JPG أو PNG لوجه أمامي واحد واضح. الدرجة ترجيح غير معاير، وليست دليلًا قطعيًا على الأصالة.")


@st.cache_resource(max_entries=1, show_spinner=False)
def load_predictor(path, checkpoint_sha256):
    # The digest invalidates the cache if a new best.pt replaces the old file.
    torch.set_num_threads(2)
    return Predictor(path, device="cpu")


checkpoint = Path(os.environ.get("FACEJPEG_CHECKPOINT", str(HERE / "best.pt")))
if not checkpoint.is_file():
    st.warning("لم يُضَف نموذج وجوه مدرّب بعد. ارفع best.pt الناتج عن تدريب REAL_CPU أو REAL إلى deploy/streamlit في GitHub.")
    st.info("حفظ الدفتر أو نشر الواجهة وحده لا يحفظ نموذجًا مدرّبًا. أوزان QUICK الهندسية لا تدعم كشف أصالة الوجوه.")
    st.stop()
try:
    with st.spinner("تحميل النموذج المحفوظ..."):
        predictor = load_predictor(str(checkpoint.resolve()), file_sha256(checkpoint))
except (ValueError, OSError, RuntimeError, EOFError, pickle.UnpicklingError, KeyError, TypeError):
    st.error("تعذّر تحميل النموذج. استخدم best.pt السليم من هذا المشروع بعد التدريب واختيار حدود التحقق؛ لا تستخدم last.pt.")
    st.stop()

enabled = predictor.face_classification_enabled
if enabled:
    st.info("تم تحميل أوزان تجربة وجوه. لا يضمن ذلك دقة معينة أو التعميم على مولّدات وصور جديدة.")
else:
    st.warning("كشف الأصالة غير متاح: الأوزان الحالية اختبار أشكال هندسية فقط. المتاح معاينة الصورة وضغط JPEG دون درجات أو أحكام أصالة.")

with st.form("compare"):
    uploaded = st.file_uploader("صورة JPG أو PNG لوجه واحد", type=["jpg", "jpeg", "png"])
    quality = st.selectbox("جودة إعادة ترميز JPEG", [100, 90, 70, 50, 30, 10], index=2)
    submitted = st.form_submit_button("تحليل الأصل والنسخة المضغوطة" if enabled else "معاينة وضغط JPEG فقط")

if not submitted:
    st.stop()
if uploaded is None:
    st.warning("ارفع صورة أولًا ثم اضغط الزر.")
    st.stop()
if uploaded.size > 15 * 1024**2:
    st.error("الملف أكبر من15 ميغابايت؛ ارفع نسخة أصغر.")
    st.stop()
try:
    with tempfile.TemporaryDirectory(prefix="facejpeg-upload-") as directory:
        path = Path(directory) / ("image" + Path(uploaded.name).suffix.lower())
        path.write_bytes(uploaded.getvalue())
        with st.spinner("معالجة الصورة ومقارنتها بالنسخة المضغوطة..."):
            original, compressed, text_original, text_compressed, note, scores_original, scores_compressed = predictor.compare_details(path, quality)
except (ValueError, OSError) as error:
    st.error(str(error))
    st.stop()

for column, image, text, scores, title in zip(
    st.columns(2), [original, compressed], [text_original, text_compressed],
    [scores_original, scores_compressed],
    ["الأصل بعد توحيد المعالجة — دون JPEG إضافي", f"النسخة المضغوطة — جودة {quality}"],
):
    with column:
        st.subheader(title)
        # PNG preserves the actual compared pixels when displaying the previews.
        st.image(image, output_format="PNG")
        st.markdown(text)
        if scores:
            st.dataframe([{"الفئة": name, "ترجيح غير معاير (%)": round(score * 100, 2)}
                          for name, score in scores.items()], hide_index=True)
st.markdown(note)
st.caption("لا تُستخدم الصور المرفوعة في التدريب. لتجربة صورة أخرى بدّل الصورة واضغط الزر؛ لا تحتاج Colab أو إعادة التدريب.")
