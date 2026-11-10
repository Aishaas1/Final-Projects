import argparse
import pickle
import gradio as gr
from PIL import Image

from facejpeg.infer import Predictor


def create_app(checkpoint, device=None):
    try:
        predictor = Predictor(checkpoint, device=device)
        startup_message = ("⚠️ كشف الأصالة غير متاح: الأوزان الحالية لم تُدرّب على صور الوجوه. "
                           "هذه واجهة معاينة وضغط JPEG فقط؛ درّب Real Training على بيانات الوجوه ثم اختر best.pt."
                           if not predictor.face_classification_enabled else
                           "تم تحميل أوزان تجربة وجوه. الدرجة غير معايرة، ولا تثبت أصالة الصورة أو التعميم على مولّدات أخرى.")
    except (ValueError, OSError, RuntimeError, EOFError, pickle.UnpicklingError, KeyError, TypeError) as exc:
        predictor = None
        startup_message = f"لا توجد أوزان صالحة للتنبؤ: {exc}"
    classify_faces = predictor is not None and predictor.face_classification_enabled

    def compare(path, quality):
        try:
            if predictor is None:
                return None, None, "", "", startup_message, None, None
            result = list(predictor.compare_details(path, int(quality)))
            for i in [5, 6]:
                result[i] = [[label, round(score * 100, 2)] for label, score in result[i].items()] if result[i] else []
            return tuple(result)
        except (ValueError, OSError) as exc:
            return None, None, "", "", f"**{exc}**", None, None

    def preview(path):
        try:
            if path:
                with Image.open(path) as im:
                    if im.width * im.height <= 25_000_000 and im.format in {"JPEG", "PNG"}:
                        im.thumbnail((512, 512))
                        return im.convert("RGB").copy()
        except (OSError, Image.DecompressionBombError):
            pass
        return None

    with gr.Blocks(title="الوجوه المولّدة وضغط JPEG", css=".gradio-container {direction: rtl;}") as demo:
        gr.Markdown("# كشف صور الوجوه المولّدة ودراسة أثر JPEG\nارفع JPG أو PNG لوجه واحد أمامي في المنتصف. الصور المعروضة موحّدة إلى 224×224 قبل المقارنة. الدرجة ترجيح للنموذج وليست دليلًا قطعيًا.")
        gr.Markdown(f"**حالة النموذج:** {startup_message}")
        upload = gr.File(label="صورة JPG أو PNG", file_types=[".jpg", ".jpeg", ".png"], type="filepath")
        raw_preview = gr.Image(label="معاينة الملف المرفوع", interactive=False, format="png")
        upload.change(preview, upload, raw_preview, api_name=False)
        quality = gr.Dropdown(choices=[100, 90, 70, 50, 30, 10], value=70, label="جودة إعادة ترميز JPEG")
        button = gr.Button("تحليل الأصل والنسخة المضغوطة" if classify_faces else "معاينة وضغط JPEG فقط — دون كشف الأصالة",
                           variant="primary", interactive=predictor is not None)
        with gr.Row():
            with gr.Column():
                original = gr.Image(label="الأصل بعد توحيد المعالجة — دون JPEG إضافي", interactive=False, format="png")
                result_original = gr.Markdown()
                scores_original = gr.Dataframe(headers=["الفئة", "ترجيح غير معاير (%)"], datatype=["str", "number"],
                    type="array", row_count=(0, "dynamic"), col_count=(2, "fixed"),
                    label="درجات الفئتين؛ الحكم في النص أعلاه", interactive=False, visible=classify_faces)
            with gr.Column():
                compressed = gr.Image(label="النسخة المضغوطة مباشرة من الأصل الموحّد", interactive=False, format="png")
                result_compressed = gr.Markdown()
                scores_compressed = gr.Dataframe(headers=["الفئة", "ترجيح غير معاير (%)"], datatype=["str", "number"],
                    type="array", row_count=(0, "dynamic"), col_count=(2, "fixed"),
                    label="درجات الفئتين بعد JPEG؛ الحكم في النص أعلاه", interactive=False, visible=classify_faces)
        note = gr.Markdown("الصور لا تُستخدم في التدريب. في Colab قد ينشئ Gradio رابطًا عامًا مؤقتًا إذا اخترت المشاركة.")
        button.click(compare, [upload, quality], [original, compressed, result_original, result_compressed, note,
                                               scores_original, scores_compressed], api_name="compare")
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/smoke/best.pt")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    create_app(a.checkpoint).launch(server_name="0.0.0.0", server_port=a.port, share=a.share)
