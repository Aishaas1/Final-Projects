"""Build a small self-contained Colab notebook: source embedded, no ZIP needed."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import nbformat


def build():
    root = Path(__file__).resolve().parents[1]
    source_files = [root / "app.py", root / "requirements.txt", root / "README_AR.md"]
    source_files += sorted((root / "facejpeg").glob("*.py"))
    source_files += [root / "scripts" / name for name in ["make_demo.py", "download_data.py"]]
    source_files += sorted((root / "docs").rglob("*.md"))
    payload = json.dumps({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
                          for p in source_files}, ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(payload, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    digest = hashlib.sha256(compressed).hexdigest()
    notebook = nbformat.read(root / "FaceJPEG_Colab.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.cells[0].source = """# كشف صور الوجوه المولّدة وأثر ضغط JPEG — نسخة مستقلة

**لا تحتاج هذه النسخة إلى تنزيل ZIP أو رفعه.** شغّل خلية التهيئة التالية، ثم المراحل1–8 لتجربة CPU سريعة، و9 لرفع صورتك و10 لواجهة Gradio.

الكود مضمن داخل الدفتر نفسه، ويُكتب في مجلد منفصل قبل تثبيت المتطلبات. **لم تُضمّن الأوزان الكبيرة أو بيانات الوجوه**؛ تنشئ التجربة السريعة أوزانها عند التشغيل. ما زال تثبيت المكتبات وتنزيل أوزان ImageNet يحتاج الإنترنت.

الإعداد الافتراضي `QUICK=True` و`SOURCE='fixture'` يختبر البرمجيات على أشكال هندسية، ولا يثبت دقة كشف وجوه مولّدة. للتدريب الحقيقي غيّر المصدر إلى kaggle أو zip أو drive في المرحلة2، واختر `QUICK=False` وGPU. تقارير التدريب البرمجي السابق وأوزانه موجودة في المستودع الكامل فقط. تُظهر الواجهة تحذيرًا ما دامت تستخدم أوزان الاختبار الهندسي."""
    bootstrap = f'''
# Embedded project source, generated from this repository; no remote source-code execution.
import base64, gzip, hashlib, json, os
from pathlib import Path
_source_blob = base64.b64decode({encoded!r})
assert hashlib.sha256(_source_blob).hexdigest() == {digest!r}, "Embedded source checksum mismatch"
_source_files = json.loads(gzip.decompress(_source_blob).decode("utf-8"))
_project_root = (Path("/content") if Path("/content").exists() else Path.cwd()) / "facejpeg_standalone"
_project_root = _project_root.resolve()
_project_root.mkdir(parents=True, exist_ok=True)
for _rel, _text in _source_files.items():
    _target = (_project_root / _rel).resolve()
    if not _target.is_relative_to(_project_root):
        raise ValueError("Unsafe embedded path")
    if _target.exists() and _target.read_text(encoding="utf-8") != _text:
        raise FileExistsError(f"{{_target}} has local changes; choose a new project directory")
    _target.parent.mkdir(parents=True, exist_ok=True)
    _target.write_text(_text, encoding="utf-8")
os.chdir(_project_root)
print("الكود جاهز، ولا حاجة إلى ZIP:", _project_root)
'''
    notebook.cells.insert(1, nbformat.v4.new_markdown_cell("## تهيئة النسخة المستقلة\nتتحقق هذه الخلية من تجزئة الكود المضمن، وتكتبه في مجلد منفصل دون استبدال تغييراتك، ثم تنتقل إليه. شغّل بعدها المرحلة1 للتثبيت."))
    notebook.cells.insert(2, nbformat.v4.new_code_cell(bootstrap.strip()))
    notebook.cells[3].source = """## 1 — تثبيت المتطلبات
الكود أصبح موجودًا بفضل خلية التهيئة؛ **لن يُطلب ZIP المشروع**. شغّل هذه الخلية للتثبيت والتحقق من PyTorch وGPU، ثم تابع المراحل بالترتيب. عند فشل التنزيل حافظ على تحقق TLS وأصلح الاتصال بدل تعطيله."""
    notebook.metadata["colab"]["name"] = "FaceJPEG_Standalone.ipynb"
    nbformat.validate(notebook)
    destination = root / "FaceJPEG_Standalone.ipynb"
    nbformat.write(notebook, destination)
    print(f"{destination.name}: {destination.stat().st_size:,} bytes; embedded files: {len(source_files)}")


if __name__ == "__main__":
    build()
