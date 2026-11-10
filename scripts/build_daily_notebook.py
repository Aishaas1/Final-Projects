"""Build an independent, single-cell daily inference notebook and Python entry point."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root)
    paths = sorted((root / "facejpeg").glob("*.py")) + [root / "app.py", root / "requirements-daily-colab.txt"]
    files = {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths}
    blob = gzip.compress(json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8"), mtime=0)
    digest = hashlib.sha256(blob).hexdigest()
    source = (root / "scripts/daily_predict_template.py").read_text(encoding="utf-8")
    source = source.replace("__DAILY_PAYLOAD__", base64.b64encode(blob).decode()).replace("__DAILY_DIGEST__", digest)
    (root / "daily_predict.py").write_text(source, encoding="utf-8")
    markdown = """# FaceJPEG — تشغيل الواجهة اليوم وغدًا، دون إعادة تدريب

هذا دفتر **تنبؤ مستقل بخلية واحدة**. يجهز كود المشروع تلقائيًا ثم يطلب `best.pt` من تدريب الوجوه الذي نفذته، أو ZIP نتائجه/النسخة الاحتياطية. لا يحتاج بيانات التدريب أو GPU أو تركيب Google Drive. ارفع **النموذج أولًا**؛ الصورة ترفعها لاحقًا داخل الواجهة.

**الخطوات:** اضغط زر تشغيل الخلية التالية وانتظر. ارفع ملف النموذج عند الطلب، ثم احتفظ بملف `facejpeg_model_backup.zip` الذي يبدأ تنزيله، وافتح رابط الواجهة الجديد. إذا كان نموذجك ما زال موجودًا في ملفات الجلسة يستعيده بدل طلب الرفع.

**لماذا عمل أمس وتعطل اليوم؟** جلسة Colab مؤقتة؛ انتهاء الجلسة قد يمحو الملفات والمكتبات والمتغيرات. حفظ الدفتر في Drive أو GitHub لا يحفظ أوزان النموذج. إذا لم تحتفظ بـ`best.pt` أو ZIP نتائجه خارج الجلسة، لا يمكن لهذا الدفتر استرجاع ملف حُذف مع انتهائها.

المقارنة تستخدم الأوزان والمعالجة وحدود القرار المحفوظة نفسها؛ هذه النسخة تصلح **بدء التشغيل والاستعادة** ولا تدّعي رفع الدقة. ترفض أوزان QUICK الهندسية بدل عرضها ككاشف وجوه. ZIP المقارنة القديمة يتيح `MODEL_CHOICE` لاختيار B=`jpeg_augmented` أو A=`baseline`؛ الأفضل رفع `best.pt` المقصود وحده لتجنب الالتباس.

رابط Gradio مؤقت ويتوقف عند توقف الخادم أو الجلسة، حتى إن لم تنته مدة الرابط القصوى. في الجلسة نفسها يمكنك اختبار صور متتابعة دون إعادة تشغيل الكود. بعد انتهاء الجلسة، افتح هذا الدفتر وشغّل الخلية وارفع النسخة المحفوظة.

إذا كانت مكتبة غير متوافقة محمّلة مسبقًا، احتفظ بالنموذج وافتح جلسة جديدة وشغّل هذا الدفتر وحده. التنبؤ على CPU لتقليل الاعتماد على GPU. الكود مضمن داخل الدفتر؛ الاتصال مطلوب لتثبيت مكتبات ناقصة وإنشاء رابط Gradio فقط. [دليل الاستخدام](https://github.com/Aishaas1/Final-Projects/blob/main/docs/DAILY_USE_AR.md).
"""
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(markdown), nbformat.v4.new_code_cell(source)])
    nb.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
                       language_info={"name": "python"}, colab={"name": "FaceJPEG_Daily_Predict.ipynb", "provenance": []})
    for index, cell in enumerate(nb.cells):
        cell["id"] = f"daily-{index}"
    nbformat.write(nb, root / "FaceJPEG_Daily_Predict.ipynb")
    print(f"Built single-cell daily inference notebook; payload sha256={digest}")


if __name__ == "__main__":
    build()
