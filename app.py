import argparse
import pickle
import gradio as gr
from PIL import Image

from facejpeg.infer import Predictor


def create_app(checkpoint):
    try:
        predictor = Predictor(checkpoint)
        startup_message = ("⚠️ لا توجد أوزان كشف وجوه حقيقية في هذه التجربة؛ الأوزان الحالية اختبار برمجيات على أشكال هندسية فقط. "
                           "درّب Real Training على بيانات الوجوه ثم اختر best.pt."
                           if predictor.bundle["data_kind"] == "procedural_fixture" else
                           "تم تحميل أوزان تجربة وجوه. الدرجة غير معايرة، ولا تثبت أصالة الصورة أو التعميم على مولّدات أخرى.")
    except (ValueError, OSError, RuntimeError, EOFError, pickle.UnpicklingError, KeyError, TypeError) as exc:
        predictor = None
        startup_message = f"لا توجد أوزان صالحة للتنبؤ: {exc}"

    def compare(path, quality):
        try:
            if predictor is None:
                return None, None, "", "", startup_message, None, None
            return predictor.compare_details(path, int(quality))
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
        button = gr.Button("تحليل الأصل والنسخة المضغوطة", variant="primary", interactive=predictor is not None)
        with gr.Row():
            with gr.Column():
                original = gr.Image(label="الأصل بعد توحيد المعالجة — دون JPEG إضافي", interactive=False, format="png")
                result_original = gr.Markdown()
                scores_original = gr.Label(label="ترجيح الفئتين — درجات غير معايرة", num_top_classes=2)
            with gr.Column():
                compressed = gr.Image(label="النسخة المضغوطة مباشرة من الأصل الموحّد", interactive=False, format="png")
                result_compressed = gr.Markdown()
                scores_compressed = gr.Label(label="ترجيح الفئتين بعد JPEG — غير معاير", num_top_classes=2)
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
