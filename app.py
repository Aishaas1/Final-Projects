import argparse
import gradio as gr

from facejpeg.infer import Predictor


def create_app(checkpoint):
    predictor = Predictor(checkpoint)

    def compare(path, quality):
        try:
            return predictor.compare(path, int(quality))
        except ValueError as exc:
            return None, None, "", "", f"**{exc}**"

    with gr.Blocks(title="الوجوه المولّدة وضغط JPEG", css=".gradio-container {direction: rtl;}") as demo:
        gr.Markdown("# كشف صور الوجوه المولّدة ودراسة أثر JPEG\nارفع JPG أو PNG لوجه واحد أمامي في المنتصف. الصور المعروضة موحّدة إلى 224×224 قبل المقارنة. الدرجة ترجيح للنموذج وليست دليلًا قطعيًا.")
        if predictor.bundle["data_kind"] == "procedural_fixture":
            gr.Markdown("## ⚠️ وضع اختبار البرمجيات فقط — الأوزان الحالية ليست كاشفًا مدرّبًا للوجوه المولّدة.")
        upload = gr.File(label="صورة JPG أو PNG", file_types=[".jpg", ".jpeg", ".png"], type="filepath")
        quality = gr.Dropdown(choices=[90, 70, 50, 30], value=70, label="جودة إعادة ترميز JPEG")
        button = gr.Button("تحليل الأصل والنسخة المضغوطة", variant="primary")
        with gr.Row():
            with gr.Column():
                original = gr.Image(label="الأصل بعد توحيد المعالجة — دون JPEG إضافي", interactive=False)
                result_original = gr.Markdown()
            with gr.Column():
                compressed = gr.Image(label="النسخة المضغوطة مباشرة من الأصل الموحّد", interactive=False)
                result_compressed = gr.Markdown()
        note = gr.Markdown("الصور لا تُستخدم في التدريب. في Colab قد ينشئ Gradio رابطًا عامًا مؤقتًا إذا اخترت المشاركة.")
        button.click(compare, [upload, quality], [original, compressed, result_original, result_compressed, note], api_name="compare")
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="artifacts/smoke/best.pt")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=7860)
    a = p.parse_args()
    create_app(a.checkpoint).launch(server_name="0.0.0.0", server_port=a.port, share=a.share)
