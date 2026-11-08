"""One self-contained, versioned notebook; reproducible source, no project ZIP prompt."""
import base64
import gzip
import hashlib
import json
import copy
from pathlib import Path

import nbformat as nbf


def configure_face_training_flow(notebook, software_test=False):
    """Invalidate stale state, handle missing data, and verify weights before UI."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    mode = 'MODE = "QUICK"  # @param ["QUICK"]' if software_test else 'MODE = "REAL_CPU"  # @param ["REAL_CPU", "REAL"]'
    code_cells[2].source = "\n".join(mode if line.startswith("MODE = ") else line
                                      for line in code_cells[2].source.splitlines())
    code_cells[2].source = ('''BEST = None
CHECKPOINTS = {}
TRAINING_READY = False
DATA_READY = False
manifest = None
if globals().get("demo") is not None:
    demo.close()
demo = None
''' + f"ALLOW_SOFTWARE_PREVIEW = {software_test!r}\n" + code_cells[2].source)

    def guarded(source, flag, message):
        return f'if not globals().get({flag!r}, False):\n    print({message!r})\nelse:\n' + "\n".join(
            "    " + line for line in source.splitlines())

    code_cells[3].source = ('''from zipfile import BadZipFile
DATA_READY = False
TRAINING_READY = False
BEST = None
CHECKPOINTS = {}
manifest = None
if globals().get("demo") is not None:
    demo.close()
demo = None
try:
''' + "\n".join("    " + line for line in code_cells[3].source.splitlines()) + '''
    DATA_READY = True
except (OSError, ValueError, RuntimeError, BadZipFile) as _data_error:
    print("⚠️ لم تجهز بيانات الوجوه؛ لن يبدأ التدريب أو التقييم أو كشف الأصالة.")
    print("السبب:", str(_data_error))
    print("للمتابعة: ارفع ZIP بيانات به real وfake من لوحة Files، ثم اختر SOURCE='zip' وحدد DATA_ZIP في المرحلة3 وأعد3 ثم4. أو أصلح الوصول إلى Kaggle. لا يوجد انتقال تلقائي إلى QUICK.")
''')
    code_cells[5].source = ("TRAINING_READY = False\nBEST = None\nCHECKPOINTS = {}\n" + guarded(
        code_cells[5].source + "\nTRAINING_READY = True", "DATA_READY",
        "لم يبدأ التدريب: أكمل تجهيز بيانات الفئتين في المرحلة4 أولًا."))
    for number in [7, 8, 9]:
        code_cells[number - 1].source = guarded(code_cells[number - 1].source, "TRAINING_READY",
            "لم تكتمل المرحلة6؛ لن نحسب نتائج اختبار أو نصدر أوزانًا غير جاهزة. أكمل تدريب الوجوه أولًا.")
    code_cells[9].source = '''import pickle
from facejpeg.infer import Predictor
from app import create_app
if globals().get("demo") is not None:
    demo.close()
def _verified_interface():
    if not globals().get("TRAINING_READY", False) or not globals().get("BEST"):
        print("الواجهة غير جاهزة للكشف: أكمل تجهيز البيانات في4 والتدريب في6. لم نحمّل أوزان QUICK بدلًا من نموذج الوجوه.")
        return None
    try:
        _checked = Predictor(BEST, "cpu")
        _faces_enabled = _checked.face_classification_enabled
        del _checked
    except (ValueError, OSError, RuntimeError, EOFError, KeyError, TypeError, pickle.UnpicklingError):
        print("تعذر تحميل best.pt صالح. أعد تحميل أوزان تجربة الوجوه السليمة أو أكمل المرحلة6.")
        return None
    if not _faces_enabled and not ALLOW_SOFTWARE_PREVIEW:
        print("رُفض فتح الواجهة بأوزان هندسية: اختر best.pt من تدريب REAL_CPU على بيانات وجوه فعلية. تغيير اسم الملف لا يكفي.")
        return None
    if not LAUNCH_UI:
        print("تشغيل الواجهة معطّل للفحص غير التفاعلي؛ اجعل LAUNCH_UI=True عند الحاجة.")
        return None
    return create_app(BEST)
demo = _verified_interface()
if demo is not None:
    demo.launch(share=SHARE_GRADIO if IN_COLAB else False, debug=False)
'''
    notebook.cells[7].source += "\n\nعند تعذر التحميل تظهر رسالة عربية وخطوات ZIP؛ تُتخطى المراحل المعتمدة على البيانات حتى إصلاح المصدر وإعادة المرحلة3 ثم4. لا يُستبدل مصدر الوجوه بعينة أشكال تلقائيًا."
    notebook.cells[19].source = """## 10 — فتح الواجهة بعد التحقق من أوزان الوجوه
تُفتح الواجهة بعد اكتمال المرحلة6 فقط. يفحص الدفتر ملف best.pt فعليًا، ويرفض الأوزان الهندسية أو الأوزان التالفة بدل استخدامها لكشف الوجوه. عند تغيير الإعدادات في المرحلة3 تُغلق الواجهة القديمة ويُلغى اختيار النموذج السابق. لذلك لا يعيد خطأ التنزيل فتح نموذج QUICK قديم.

بعد التدريب ارفع JPG/PNG لوجه أمامي واحد واضح، واختر جودة JPEG. كل صورة جديدة تحتاج زر التحليل فقط. الرابط مؤقت مع جلسة Colab. للاستعمال لاحقًا احتفظ بـbest.pt وافتح دفتر التنبؤ فقط. في دفتر اختبار البرمجيات المنفصل يسمح بعرض JPEG دون أحكام أصالة، وموسوم بذلك بوضوح."""


def build_jpeg_notebook(root, source):
    notebook = copy.deepcopy(source)
    notebook.metadata["colab"]["name"] = "FaceJPEG_JPEG_Training.ipynb"
    notebook.cells[0].source = """# FaceJPEG 2.3.1 — تدريب الوجوه على CPU ومقارنة JPEG
الافتراضي الآن MODE=REAL_CPU؛ لا حاجة لتعديله من QUICK. يعمل دون GPU ودون تركيب Drive افتراضيًا. يحتاج تنزيل بيانات وجوه فوتوغرافية ومولدة عبر Kaggle أو ZIP/Drive. الكود مستقل ومضمن، لكن لا توجد أوزان كشف وجوه مدربة مضمنة، ولا نعد بدقة معينة. لا تُفتح واجهة الكشف قبل اكتمال التدريب وتحميل أوزان وجوه صالحة.

التجربة A تدرب نموذجًا أساسيًا، وB تستخدم JPEG عشوائيًا لصور التدريب فقط باحتمال0.5 وجودات90/70/50/30. نفس التقسيم والبذرة وإعدادات التدريب؛ التحقق يبقى أصليًا. يُختار أفضل نموذج وحدوده لكل تجربة من التحقق، ويُثبت النموذجان قبل فتح الاختبار. كل جودة اختبار مستقلة من الأصل، وحدود كل نموذج ثابتة عبر الجودات.

للتجربة البرمجية السريعة فقط يوجد دفتر منفصل FaceJPEG_JPEG_Smoke_Test.ipynb؛ أوزانه هندسية وليست كاشف وجوه. اقرأ قبل كل مرحلة، واحتفظ بـbest.pt والنتائج على جهازك أو Drive قبل انتهاء الجلسة. USE_DRIVE=False افتراضيًا لتجنب اعتماد التشغيل على مصادقة Drive؛ فعّله إن رغبت في الحفظ الدائم. تغيير MODE أو اسم الوزن وحده لا يدرّب الوجوه."""
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    def stage(number, markdown=None, code=None):
        if markdown is not None:
            notebook.cells[2 * number - 1].source = markdown.strip()
        if code is not None:
            code_cells[number - 1].source = code.strip()

    code_cells[2].source = "\n".join(
        'EXPERIMENT = "jpeg_quick_v2_3_1" if MODE == "QUICK" else "jpeg_faces_cpu_v2_3_1" if MODE == "REAL_CPU" else "jpeg_faces_gpu_v2_3_1"'
        if line.startswith("EXPERIMENT = ") else line for line in code_cells[2].source.splitlines()
    ) + '\nJPEG_TRAIN_QUALITIES = [90, 70, 50, 30]\nJPEG_PROBABILITY = 0.5\nMODEL_FOR_UI = "jpeg_augmented"  # أو baseline؛ اختيار محدد مسبقًا، لا فائز تلقائيًا\n'
    stage(3, markdown="""## 3 — الإعدادات ومصدر البيانات والحفظ
الافتراضي REAL_CPU لتدريب الوجوه بلا GPU، وREAL لضبط آخر طبقات الشبكة على GPU. اختبار QUICK له دفتر منفصل. اختر المصدر kaggle أو ZIP أو Drive. ضع ZIP البيانات في لوحة Files واكتب مساره في DATA_ZIP، أو جهّز مجلد real وfake على Drive. اترك USE_DRIVE=False إذا تعذرت مصادقته، ونزّل النتائج لاحقًا من المرحلة9.

JPEG_PROBABILITY=0.5 يعني أن كل زيارة لصورة تدريب تختار نسخة JPEG باحتمال50%، وتختار جودة عشوائية بالتساوي من القائمة. البقية تستخدم الأصل؛ القلب الأفقي له السياسة نفسها في A وB. على CPU تُستخرج ميزات الأصل والنسخ المضغوطة مرة واحدة ثم تُختار عشوائيًا كل حقبة. تغيير جودة الواجهة وحده لا يغيّر التدريب.

في REAL_CPU الحد20 حقبة وحتى1000 صورة لكل فئة بعد التدقيق والتقسيم. هذه عينة ابتدائية لا ضمان دقة. اختر اسم EXPERIMENT جديدًا عند تغيير أي إعداد أو البيانات؛ تُقفل خطة المقارنة لمنع تعديلها بناء على الاختبار. يبدأ الحفظ المؤقت في RUN؛ لا يغني حفظ الدفتر عن حفظ الأوزان.""")
    stage(6, markdown="""## 6 — تدريب A وB وتثبيتهما قبل الاختبار
يُدرّب A أولًا ثم B على manifest نفسه والبذرة وإعدادات التدريب نفسها. B فقط تضيف JPEG. لا تُضغط صور التحقق أثناء التدريب، ولا تُستخدم صور الاختبار لاستخراج الميزات. كل نسخة تدريب مضغوطة مستقلة من الأصل الموحد224×224.

على CPU قد يستغرق استخراج ميزات B وقتًا أطول لأنه يحسب أربع جودات وقلبها الأفقي؛ لا يعيد الشبكة في كل حقبة. التوقف المبكر واختيار best.pt وحدود القرار من التحقق فقط. comparison_plan.json يحفظ الخطة، وmodels_locked.json يسجل بصمتي النموذجين وحدودهما قبل أي تقييم اختبار. إعادة الخلية بعد اكتمالها تعيد استخدام النموذجين المثبتين. بعد انقطاع جزئي يُستكمل التدريب بنفس الإعدادات.""", code='''
from facejpeg.experiment import fit_jpeg_comparison
_cached_cpu = QUICK or CPU_REAL
CHECKPOINTS = fit_jpeg_comparison(
    PREPARED, RUN, jpeg_qualities=JPEG_TRAIN_QUALITIES, jpeg_probability=JPEG_PROBABILITY,
    epochs=EPOCHS, batch_size=BATCH_SIZE, quick=QUICK, resume=RESUME, workers=WORKERS,
    allow_cpu_real=ALLOW_CPU_REAL or CPU_REAL, cache_features=_cached_cpu,
    warmup_epochs=0 if _cached_cpu else 1, finetune_blocks=0 if _cached_cpu else 3,
    use_amp=not _cached_cpu, lr=1e-3 if _cached_cpu else 1e-4,
    use_scheduler=_cached_cpu, early_stopping_patience=8 if _cached_cpu else 5)
assert MODEL_FOR_UI in CHECKPOINTS
BEST = CHECKPOINTS[MODEL_FOR_UI]
for _name in CHECKPOINTS:
    print("Training:", _name)
    display(pd.read_csv(RUN / _name / "history.csv"))
    display(DisplayImage(filename=str(RUN / _name / "learning_curves.png")))
print("النموذجان مثبتان، ولم نفتح الاختبار بعد.")
''')
    stage(7, markdown="""## 7 — التحقق من إعادة التحميل قبل الاختبار
نحمّل كل best.pt مرتين ونفحص تطابق درجة صورة تحقق واحدة. هذا اختبار اتساق برمجي، وليس حساب دقة وجوه. تظهر بصمات النموذجين وحدودهما المحفوظة. لا توجد تنبؤات اختبار في هذه الخلية.""", code='''
from facejpeg.infer import Predictor
from facejpeg.common import read_rgb, file_sha256
_example = PREPARED / manifest.loc[manifest.split == "val", "path"].iloc[0]
for _name, _checkpoint in CHECKPOINTS.items():
    _first = Predictor(_checkpoint, "cpu")
    _second = Predictor(_checkpoint, "cpu")
    assert abs(_first.score(read_rgb(_example)) - _second.score(read_rgb(_example))) < 1e-7
    print(_name, "SHA256:", file_sha256(_checkpoint), "thresholds:", _first.bundle["thresholds"])
    del _first, _second
print("نجح اختبار الحفظ وإعادة التحميل للنموذجين.")
''')
    stage(8, markdown="""## 8 — اختبار النموذجين ومقارنة أثر JPEG
نتأكد من أن بصمتي النموذجين لم تتغيرا، ثم نقيّم كليهما على صور الاختبار نفسها: الأصل ثم100/90/70/50/30/10، كل نسخة مباشرة من الأصل الموحد. تُستخدم الحدود المحفوظة لكل نموذج دون تعديلها بين الجودات.

مقام الاكتشاف جميع المولّدات، والاتهام الخاطئ جميع الفوتوغرافيات، وغير الحاسم جميع الاختبار. comparison.csv وcomparison.png وCOMPARISON_AR.md تعرض A وB والفروق بالنقاط المئوية. داخل كل تجربة مصفوفات الالتباس وROC وأمثلة تغيّر الحكم إن وجدت. لا نعلن تحسنًا لمجرد ارتفاع درجة صورة، ولا نفسر QUICK كدقة وجوه. لا نعدّل التدريب بعد رؤية هذه النتائج.""", code='''
from facejpeg.experiment import evaluate_jpeg_comparison
table = evaluate_jpeg_comparison(PREPARED, RUN, device="cpu" if QUICK or CPU_REAL else None)
display(table[["quality", "n_fake", "baseline_fake_detected", "jpeg_augmented_fake_detected",
               "n_real", "baseline_real_falsely_accused", "jpeg_augmented_real_falsely_accused",
               "n_total", "baseline_abstain_total", "jpeg_augmented_abstain_total"]])
display(DisplayImage(filename=str(RUN / "comparison.png")))
for _name in CHECKPOINTS:
    for _figure in ["confusion_matrices.png", "roc_curves.png", "changed_examples.png"]:
        _path = RUN / _name / "evaluation" / _figure
        if _path.exists():
            print(_name, _figure)
            display(DisplayImage(filename=str(_path)))
display(Markdown((RUN / "COMPARISON_AR.md").read_text()))
''')
    code_cells[8].source = code_cells[8].source.replace(
        'if p.is_file() and (EXPORT_INCLUDE_PREPARED',
        'if p.is_file() and "feature_cache" not in p.relative_to(RUN).parts and (EXPORT_INCLUDE_PREPARED',
    )
    stage(9, markdown="""## 9 — حفظ النموذجين والتقارير وتنزيل ZIP النتائج
يتضمن ZIP أفضل وlast.pt لكل نموذج والخطة والبصمات والتقارير وmanifest. يستبعد ميزات CPU القابلة لإعادة الحساب، وصور prepared افتراضيًا لتقليل زمن الضغط. لاستكمال التدريب احتفظ بمجلد RUN كاملًا على Drive، أو فعّل EXPORT_INCLUDE_PREPARED=True للاحتفاظ بصور الاستكمال في ZIP.

بعد اكتمال الخلية نزّل ZIP من لوحة Files في Colab؛ لا يُنزَّل تلقائيًا. للتنبؤ لاحقًا استخدم best.pt من تجربة وجوه فعلية (داخل baseline أو jpeg_augmented)، واحتفظ بمعالجة الصورة والحدود المحفوظة. حفظ الدفتر وحده لا يحفظ الأوزان.""")
    smoke = copy.deepcopy(notebook)
    smoke.metadata["colab"]["name"] = "FaceJPEG_JPEG_Smoke_Test.ipynb"
    smoke.cells[0].source = """# اختبار برمجيات JPEG فقط — ليس كاشف وجوه
الافتراضي QUICK: أشكال هندسية للتأكد من التدريب والحفظ والاستكمال والمقارنة. لا تستخدم درجاته أو أوزانه لكشف أصالة وجوه. الواجهة تسمح بمعاينة JPEG دون أحكام أصالة. لتدريب الوجوه استخدم FaceJPEG_Real_CPU_Ready.ipynb الذي يبدأ بـREAL_CPU ويتطلب بيانات وجوه فعلية."""
    configure_face_training_flow(smoke, software_test=True)
    nbf.validate(smoke)
    nbf.write(smoke, root / "FaceJPEG_JPEG_Smoke_Test.ipynb")
    configure_face_training_flow(notebook)
    nbf.validate(notebook)
    nbf.write(notebook, root / "FaceJPEG_JPEG_Training.ipynb")
    notebook.metadata["colab"]["name"] = "FaceJPEG_Real_CPU_Ready.ipynb"
    nbf.write(notebook, root / "FaceJPEG_Real_CPU_Ready.ipynb")


def build():
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in ["app.py", "requirements.txt", "requirements-colab.txt", "requirements-predict-colab.txt"]]
    paths += sorted((root / "facejpeg").glob("*.py"))
    paths += [root / "scripts" / name for name in ["make_demo.py", "download_data.py"]]
    paths += [root / "docs" / name for name in ["DATASET_CARD_AR.md", "PRESENTATION_AR.md", "JPEG_TRAINING_AR.md"]]
    payload = json.dumps({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths}, ensure_ascii=False).encode()
    blob = gzip.compress(payload, mtime=0)
    encoded, digest = base64.b64encode(blob).decode(), hashlib.sha256(blob).hexdigest()
    nb = nbf.v4.new_notebook()
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(s.strip()))

    def code(s):
        cells.append(nbf.v4.new_code_cell(s.strip()))

    md("""# FaceJPEG — كشف الوجوه المولّدة ودراسة أثر JPEG
## دفتر نهائي مستقل، الإصدار 2.3.1
الكود مضمن هنا؛ **لا تحتاج ZIP المشروع**. الافتراضي REAL_CPU لتدريب الوجوه بلا GPU. تحتاج بيانات وجوه فوتوغرافية ومولدة؛ لا توجد أوزان كشف وجوه مدربة مضمنة. يتوفر اختبار QUICK الهندسي في دفتر منفصل FaceJPEG_JPEG_Smoke_Test.ipynb فقط لغرض التحقق البرمجي.

حدّد مصدر البيانات واسم تجربة جديدًا في المرحلة3، ثم شغّل المراحل بالترتيب أو Run all. الافتراضي Kaggle؛ ZIP البيانات بديل يُرفع من لوحة Files. USE_DRIVE=False افتراضيًا، ويمكن تفعيله للحفظ إن نجحت المصادقة. لتجربة JPEG بنموذجين استخدم FaceJPEG_Real_CPU_Ready.ipynb.

كل مرحلة تعرض تقدمها. `events.log` و`status.json` يسجلان المرحلة وآخر عملية وETA من القياسات؛ استمرار heartbeat يثبت أن المراقب حي فقط، ولا يثبت تقدم العداد أو سلامة كل عملية. لا توجد مهلة تلغي تدريبًا سليمًا. حدود القرار تختار من التحقق قبل فتح الاختبار، والدرجة ترجيح غير معاير وليست شهادة أصالة.""")
    md("""## 1 — تهيئة الكود وفحص بيئة التشغيل وGPU
تكتب هذه الخلية الكود المضمن في مجلد إصدار مستقل، وتحافظ على أي تغييرات محلية. ثم تفحص Torch/Torchvision وCUDA واسم البطاقة وذاكرتها. `+cpu` يصف بناء PyTorch؛ و`GPU=False` وحده لا يثبت سبب بطء جلسة سابقة. عند تغيير إصدار المشروع استخدم مجلدًا جديدًا بدل الكتابة فوق تعديلاتك.""")
    bootstrap = f'''
import os, sys, json, gzip, base64, hashlib, importlib.util, subprocess, time
from pathlib import Path
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
_blob = base64.b64decode({encoded!r})
assert hashlib.sha256(_blob).hexdigest() == {digest!r}, "Embedded source checksum failed"
_files = json.loads(gzip.decompress(_blob).decode("utf-8"))
_base = Path(os.environ.get("FACEJPEG_WORKSPACE", "/content" if IN_COLAB else str(Path.cwd())))
PROJECT = (_base / "facejpeg_final_v2_3_1").resolve()
PROJECT.mkdir(parents=True, exist_ok=True)
for _rel, _text in _files.items():
    _path = (PROJECT / _rel).resolve()
    if not _path.is_relative_to(PROJECT):
        raise ValueError("Unsafe source path")
    if _path.exists() and _path.read_text(encoding="utf-8") != _text:
        raise FileExistsError(f"{{_path}} has local changes: choose a new FACEJPEG_WORKSPACE or retain those changes")
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(_text, encoding="utf-8")
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT)) if str(PROJECT) not in sys.path else None
def _refresh_project_imports():
    # Reload only this project's modules if an earlier notebook used another folder.
    removed = []
    for _name, _module in list(sys.modules.items()):
        if _name == "facejpeg" or _name.startswith("facejpeg.") or _name in {{"app", "scripts.make_demo", "scripts.download_data"}}:
            _file = getattr(_module, "__file__", None)
            if not _file or not Path(_file).resolve().is_relative_to(PROJECT):
                sys.modules.pop(_name, None)
                removed.append(_name)
    return removed
_removed = _refresh_project_imports()
if _removed:
    print("أعيد تهيئة وحدات المشروع من النسخة الجديدة:", len(_removed))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / ".cache/matplotlib"))
_torch_pair_ok = False
try:
    import torch, torchvision
    _torch_pair_ok = True
    print("PyTorch:", torch.__version__, "Torchvision:", torchvision.__version__)
    print("CUDA build:", torch.version.cuda, "GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        _free, _total = torch.cuda.mem_get_info()
        print("GPU:", torch.cuda.get_device_name(), "free/total GiB:", round(_free/2**30,2), round(_total/2**30,2))
    else:
        print("لا توجد CUDA متاحة؛ يمكن تدريب الوجوه بوضع REAL_CPU. QUICK اختبار أشكال فقط، وREAL يحتاج GPU.")
except ImportError:
    print("Torch/Torchvision غير مثبتين؛ ستُعرض تعليمات التثبيت في المرحلة2.")
except Exception as _error:
    print("فشل استيراد زوج Torch/Torchvision:", type(_error).__name__)
print("Project:", PROJECT)
'''
    code(bootstrap)
    md("""## 2 — تثبيت المكتبات مع الحفاظ على CUDA
في Colab نحافظ على زوج Torch/Torchvision المثبت بدل إعادة تثبيته بعجلة CPU. تُثبت بقية المكتبات بنطاقات متوافقة وتظهر رسائل pip مع نبضة كل15 ثانية أثناء الانتظار. لا تعني النبضة أن التنزيل تقدم؛ راجع آخر سطر pip. عند خطأ استيراد زوج Torch، ابدأ جلسة Colab نظيفة مع GPU؛ لا تُعطّل التحقق من TLS. محليًا يُستخدم requirements.txt إذا كان Torch غائبًا.""")
    code('''
import threading
if IN_COLAB and not _torch_pair_ok:
    raise RuntimeError("ابدأ جلسة Colab نظيفة مع CPU أو GPU حسب النمط وأعد التشغيل؛ لن نستبدل زوج Torch المعطوب تلقائيًا.")
if _torch_pair_ok:
    import re
    _torch_version = tuple(map(int, re.match(r"([0-9]+)[.]([0-9]+)", str(torch.__version__)).groups()))
    if not (2,5) <= _torch_version < (3,0):
        raise RuntimeError("يلزم Torch >=2.5 و<3 مع Torchvision المتوافق؛ اختر إصدارًا مدعومًا من تعليمات PyTorch الرسمية أو جلسة Colab حديثة.")
_req = "requirements-colab.txt" if _torch_pair_ok else "requirements.txt"
_done = threading.Event()
_install_start = time.monotonic()
def _install_heartbeat():
    while not _done.wait(15):
        print(f"[تثبيت المتطلبات] ما زالت عملية pip تعمل؛ elapsed={time.monotonic()-_install_start:.0f}s — راجع آخر عملية تنزيل أعلاه", flush=True)
_thread = threading.Thread(target=_install_heartbeat, daemon=True)
_thread.start()
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", _req], check=True)
finally:
    _done.set(); _thread.join(timeout=1)
import torch, torchvision
from facejpeg.runtime import hardware_info
from IPython.display import display, Markdown, Image as DisplayImage
print(json.dumps(hardware_info(), ensure_ascii=False, indent=2))
print(f"انتهى التثبيت والتحقق خلال {time.monotonic()-_install_start:.1f}s")
''')
    md("""## 3 — اختيار Quick Test أو Real Training والحفظ
QUICK يفحص البرمجيات على40 شكلًا. REAL يدرّب الوجوه على GPU مع ضبط آخر3 كتل. REAL_CPU يدرّب الرأس على صور وجوه فعلية بلا GPU: يثبت الشبكة ويخزن ميزات التدريب والتحقق مرة واحدة، دون صور الاختبار. يستخدم عينة حتى1000 صورة لكل فئة بعد التدقيق والتقسيم و20 حقبة كحد أقصى وجدولة للتعلم وإيقافًا مبكرًا. هذه استراتيجية عملية لغياب GPU، ولا تضمن نسبة دقة أو تكافئ ضبط الشبكة كاملة. اختر اسم تجربة جديدًا عند تغيير البيانات أو الاستراتيجية، والإعدادات نفسها للاستكمال.

`DATA_ZIP` مسار ملف سبق رفعه من شريط ملفات Colab، لا رابط ويب. `SOURCE='drive'` يقرأ مجلدًا مستخرجًا بـreal وfake. تفعيل USE_DRIVE يحفظ الصور المجهزة وbest.pt وlast.pt والتقارير. لا تضع أسرارًا في الخلايا. Batch Size يُختار تحفظيًا من الذاكرة، ويمكن تعديله عند الحاجة؛ ليس ضمانًا ضد نفاد الذاكرة.""")
    code('''
MODE = "REAL_CPU"  # @param ["REAL_CPU", "REAL"]
SOURCE = "kaggle"  # @param ["kaggle", "zip", "drive"]
EXPERIMENT = "quick_v2_3" if MODE == "QUICK" else "faces_cpu_v2_3" if MODE == "REAL_CPU" else "faces_gpu_v2_3"
USE_DRIVE = False  # @param {type:"boolean"}
DATA_ZIP = ""  # مثال /content/faces.zip
DRIVE_RAW_PATH = "/content/drive/MyDrive/facejpeg/raw"
METADATA_CSV = None  # اختياري: path,identity,source_id بمسارات نسبية
EPOCHS = 2 if MODE == "QUICK" else 20 if MODE == "REAL_CPU" else 10
BATCH_SIZE = 64 if MODE == "REAL_CPU" else None  # دفعة ميزات، وليست64 صورة للشبكة
WORKERS = 0 if MODE == "REAL_CPU" else None
MAX_PER_CLASS = 1000 if MODE == "REAL_CPU" else None
RESUME = True
ALLOW_CPU_REAL = False  # سماح صريح لتدريب الوجوه البطيء بلا GPU
EXPORT_INCLUDE_PREPARED = False  # ZIP خفيف؛ صور الاستكمال محفوظة في Drive عند تفعيله
SHARE_GRADIO = True  # رابط عام مؤقت عند تشغيل الواجهة في Colab
LAUNCH_UI = os.environ.get("FACEJPEG_NO_UI") != "1"
assert MODE in {"QUICK", "REAL", "REAL_CPU"} and SOURCE in {"kaggle", "zip", "drive"}
QUICK = MODE == "QUICK"
CPU_REAL = MODE == "REAL_CPU"
if MODE == "REAL" and not torch.cuda.is_available() and not ALLOW_CPU_REAL:
    raise RuntimeError("اختر MODE='REAL_CPU' لتدريب الوجوه بلا GPU، أو فعّل GPU لوضع REAL. QUICK اختبار برمجيات فقط.")
if USE_DRIVE or (not QUICK and SOURCE == "drive"):
    if not IN_COLAB:
        raise RuntimeError("محليًا استخدم ZIP/مجلدًا محليًا بدل تركيب Drive")
    from google.colab import drive
    drive.mount("/content/drive")
RUNS_BASE = Path("/content/drive/MyDrive/facejpeg_runs") if USE_DRIVE else PROJECT / "runs"
RUN = RUNS_BASE / EXPERIMENT
PREPARED = RUN / "prepared"
RUN.mkdir(parents=True, exist_ok=True)
print("MODE:", MODE, "data:", "procedural fixture — software test only" if QUICK else SOURCE)
print("RUN:", RUN)
''')
    md("""## 4 — تحميل البيانات وفحصها وتقسيمها
REAL يقترح [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces): فوتوغرافيات FFHQ ووجوه StyleGAN. نحو4GiB ميزانية تخطيط للتنزيل لا قياسًا تحققنا منه؛ اترك12–20GiB. راجع `docs/DATASET_CARD_AR.md` وشروط المصدر غير التجاري والاعتمادات؛ ترخيص مرآة Kaggle الحالي لم يمكن فحصه بسبب الحجب السابق. لا تتضمن هذه الحزمة بيانات الوجوه.

403 من الوكيل يختلف عن401 للمصادقة و403 لصلاحية المجموعة و404 لعدم وجودها؛ تظهر رسالة تشخيصية ولا نفترض نجاح التنزيل. ZIP/Drive بديلان. الهيكل `raw/real/a.jpg` و`raw/fake/b.png`؛ يقبل تداخل train/valid/test لكنه يعيد تقسيمه بعد التدقيق.

تُفحص كل الصور والتكرارات أولًا. جميع النسخ/الهوية المعروفة تبقى في مجموعة واحدة؛ تقسيم مجموعات طبقي تقريبي70/15/15. تُستبعد تعارضات التصنيف وتُحذف التكرارات المتطابقة. لا يضمن pHash كشف كل النسخ أو فصل الأشخاص بلا metadata. بعد التقسيم تُطبَّق المعالجة نفسها RGB/قص224×224/PNG للفئتين؛ لا تزال بصمات المصدر والضغط السابق قيدًا. يظهر تقدم audit ثم grouping ثم normalization منفصلًا. عند وجود manifest محفوظ يُفحص ويُعاد استعماله دون إعادة تقسيم.""")
    code('''
import pandas as pd
from facejpeg.data import prepare, safe_extract, validate_manifest
from scripts.make_demo import make_demo
from scripts.download_data import download
_kind = "procedural_fixture" if QUICK else "research_faces"
if not (PREPARED / "manifest.csv").exists():
    if QUICK:
        RAW_ROOT = PROJECT / "data" / (EXPERIMENT + "_fixture")
        if not RAW_ROOT.exists():
            make_demo(RAW_ROOT, count=20)
    elif SOURCE == "kaggle":
        RAW_ROOT = download(RUN / "download.json")
    elif SOURCE == "zip":
        if not DATA_ZIP or not Path(DATA_ZIP).is_file():
            raise FileNotFoundError("ارفع ZIP البيانات من أيقونة الملفات في Colab ثم اكتب مساره في DATA_ZIP بالمرحلة3.")
        RAW_ROOT = safe_extract(DATA_ZIP, PROJECT / "data" / (EXPERIMENT + "_raw"))
    else:
        RAW_ROOT = Path(DRIVE_RAW_PATH)
    prepare(RAW_ROOT, PREPARED, metadata=METADATA_CSV, data_kind=_kind, max_per_class=MAX_PER_CLASS)
manifest = validate_manifest(PREPARED)
audit = json.loads((PREPARED / "audit.json").read_text())
if audit["data_kind"] != _kind:
    raise ValueError("هذا الاسم مستخدم لبيانات/وضع مختلف. اختر اسم تجربة جديدًا.")
display(pd.crosstab(manifest.split, manifest.label).rename(columns={0:"فوتوغرافية/رمز0",1:"مولّدة/رمز1"}))
display(pd.read_csv(PREPARED / "source_format_resolution.csv"))
print("تم فحص التقسيم: لا تداخل في المجموعات أو الهوية/المصدر المعروفين.")
''')
    md("""## 5 — بناء MobileNetV3-Small
نفحص البنية ورأسًا ثنائيًا (real=0, fake=1). أوزان ImageNet الرسمية تُحمّل مع فحص التجزئة في المرحلة6. REAL_CPU يثبت ميزات ImageNet ويدرّب الرأس على الوجوه، وREAL يضبط آخر3 كتل. الأوزان الهندسية لا تفعّل الكشف.""")
    code('''
from facejpeg.common import build_model
_probe = build_model(pretrained=False)
print("Architecture: MobileNetV3-Small; outputs:", _probe.classifier[-1].out_features)
print("Parameters:", sum(p.numel() for p in _probe.parameters()))
del _probe
''')
    md("""## 6 — التدريب مع Batch/Epoch وAMP والاستكمال
يظهر Loss وAccuracy أثناء كل دفعة، ووقت الحقبة وETA من أدائها الفعلي، ويكتب monitor نبضة كل10ثوانٍ. تستخدم CUDA نقلًا مثبتًا وعمّال تحميل مناسبين، وAMP/GradScaler فقط على CUDA؛ التحقق والتنبؤ FP32 للحفاظ على الاتساق. Loss التدريب موزون بعدد الفئات، والتحقق BCE غير موزون؛ Accuracy التدريب online عند0.5.

يُحفظ الأفضل بأقل Validation Loss، وlast.pt مع المحسن وGradScaler والعشوائية وحالة جدولة التعلم. CPU_REAL يستخرج الميزات مرة واحدة ثم يدرّب الرأس؛ لا يعاد حساب الشبكة في كل حقبة. يخفض معدل التعلم عند توقف تحسن التحقق، ويوقف بعد8 حقبات بلا تحسن (5 في المسار العادي). بعد الانقطاع تُعاد الحقبة غير المكتملة. احتفظ بملفي الأوزان وprepared/ والإعدادات نفسها. الحدود من التحقق فقط قبل الاختبار، ولا توجد مهلة تلغي تدريبًا سليمًا.""")
    code('''
from facejpeg.train import train
BEST = train(PREPARED, RUN, epochs=EPOCHS, batch_size=BATCH_SIZE, quick=QUICK,
             resume=RESUME, workers=WORKERS, allow_cpu_real=ALLOW_CPU_REAL or CPU_REAL,
             warmup_epochs=0 if CPU_REAL else 1, finetune_blocks=0 if CPU_REAL else 3,
             use_amp=not CPU_REAL, cache_features=CPU_REAL, lr=1e-3 if CPU_REAL else 1e-4,
             use_scheduler=CPU_REAL, early_stopping_patience=8 if CPU_REAL else 5)
display(pd.read_csv(RUN / "history.csv"))
display(DisplayImage(filename=str(RUN / "learning_curves.png")))
print("Measured timing:", (RUN / "timing.json").read_text())
''')
    md("""## 7 — إعادة التحميل والتقييم الأساسي
تُحمّل الأوزان مرتين ونقارن درجة الصورة نفسها من التحقق. ثم نفتح الاختبار الأصلي فقط، دون تغيير النموذج أو الحدود. Precision/Recall/F1/Accuracy الثنائية عند0.5 مقاييس مساعدة مع zero_division=0؛ حكم الواجهة ثلاثي وله تغطية وغير حاسم ومقامات مستقلة. لا تُفسّر أرقام Quick على أنها نتائج وجوه.""")
    code('''
from facejpeg.infer import Predictor
from facejpeg.common import read_rgb
from facejpeg.evaluate import evaluate
predictor = Predictor(BEST, "cpu" if QUICK or CPU_REAL else None)
_second = Predictor(BEST, "cpu" if QUICK or CPU_REAL else None)
_example = PREPARED / manifest.loc[manifest.split == "val", "path"].iloc[0]
assert abs(predictor.score(read_rgb(_example)) - _second.score(read_rgb(_example))) < 1e-7
del _second
baseline = evaluate(PREPARED, BEST, RUN / "baseline", device="cpu" if QUICK or CPU_REAL else None, qualities=["original"])
display(baseline[["quality","binary_accuracy_at_0_5_all","binary_precision_at_0_5","binary_recall_at_0_5","binary_f1_at_0_5","roc_auc_all"]])
print("ثبّتت الأوزان والحدود؛ لن نختارها من نتائج الاختبار التالية.")
''')
    md("""## 8 — تجربة JPEG المستقلة والتقرير العربي
كل صورة اختبار في الأصل ثم100/90/70/50/30/10، مباشرة من الأصل الموحّد224×224؛ لا ضغط متسلسل ولا تسرب نسخ إلى التدريب. original ليس Q100؛ حتىQ100 إعادة ترميز خسِرة، وقد يكون الأصل مضغوطًا سابقًا. المقارنة تخص هذا البروتوكول عند224×224، لا الدقة الخام كاملة.

اكتشاف المولّد مقامه **كل المولّدات**؛ الاتهام الخاطئ مقامه **كل الفوتوغرافيات**؛ غير الحاسم مقامه **كل الاختبار**. مصفوفة2×3 تُظهره. تُحفظ ROC/AUC وF1 والتغير بالنقاط المئوية وأمثلة الأحكام المتبدلة إن وجدت، وتقرير عربي من الأرقام نفسها. عدم وجود مثال متغير يُذكر بدل اختلاقه.""")
    code('''
RESULTS = RUN / "evaluation"
table = evaluate(PREPARED, BEST, RESULTS, device="cpu" if QUICK or CPU_REAL else None)
display(table)
for _figure in ["jpeg_effect.png", "confusion_matrices.png", "roc_curves.png", "changed_examples.png"]:
    if (RESULTS / _figure).exists():
        display(DisplayImage(filename=str(RESULTS / _figure)))
display(Markdown((RESULTS / "REPORT_AR.md").read_text()))
''')
    md("""## 9 — حفظ النموذج والتقارير وتنزيل النتائج
أفضل أوزان وlast.pt والسجلات محفوظة بالفعل في RUN؛ وفي Google Drive إن فُعّل USE_DRIVE. ZIP التالي يتضمنها مع ملفات التقسيم والتقارير؛ لا يتضمن صور التدريب افتراضيًا لتجنب انتظار ضغط آلاف الصور. **استكمال التدريب يحتاج صور prepared أيضًا**: احتفظ بمجلد Drive كاملًا أو اختر EXPORT_INCLUDE_PREPARED=True قبل التصدير. تنزيل ZIP يدويًا من لوحة Files لا يعطل Run all، ولا يُفتح مربع رفع جديد.""")
    code('''
import zipfile
RESULT_ZIP = PROJECT / (EXPERIMENT + "_results.zip")
_export_files = [p for p in RUN.rglob("*") if p.is_file() and (EXPORT_INCLUDE_PREPARED or not p.is_relative_to(PREPARED / "images"))]
from tqdm.auto import tqdm
with zipfile.ZipFile(RESULT_ZIP, "w", zipfile.ZIP_DEFLATED) as _zip:
    for _p in tqdm(_export_files, desc="Export results", unit="file"):
        _zip.write(_p, _p.relative_to(RUN))
print("Results ZIP:", RESULT_ZIP)
print("Best checkpoint:", BEST)
print("Resume images included:", EXPORT_INCLUDE_PREPARED)
''')
    md("""## 10 — واجهة Gradio العربية
ارفع JPG/PNG لوجه أمامي واحد واضح في المنتصف. في QUICK تعرض الواجهة الأصل الموحّد والنسخة المضغوطة مع عبارة «كشف الأصالة غير متاح»؛ لا تُحسب درجات Real/AI ولا تظهر أحكام عليها. بعد تدريب صور الوجوه الفعلية وتحميل best.pt تظهر الدرجات والحكم الثلاثي، وتبقى الدرجات غير معايرة. لا وجه/عدة وجوه/ملف تالف تظهر برسالة عربية. Haar بوابة بسيطة قد تخطئ.

في Colab ينشئ SHARE_GRADIO رابطًا عامًا مؤقتًا؛ افتحه بعد تشغيل الخلية ولا ترفع صورًا حساسة. محليًا لا حاجة للمشاركة العامة: `python app.py --checkpoint /path/to/best.pt`. أغلق الواجهة بعد الانتهاء.""")
    code('''
from app import create_app
if LAUNCH_UI:
    if globals().get("demo") is not None:
        demo.close()
    demo = create_app(BEST)
    demo.launch(share=SHARE_GRADIO if IN_COLAB else False, debug=False)
else:
    print("تشغيل الواجهة معطّل لهذا الفحص غير التفاعلي؛ شغّل create_app(BEST).launch() عند الحاجة.")
''')
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"},
                   "colab": {"name": "FaceJPEG_Final_Colab.ipynb", "provenance": []}}
    raw_notebook = copy.deepcopy(nb)
    configure_face_training_flow(nb)
    nbf.validate(nb)
    for name in ["FaceJPEG_Final_Colab.ipynb", "FaceJPEG_Standalone.ipynb", "FaceJPEG_Colab.ipynb"]:
        nbf.write(nb, root / name)
    cpu = copy.deepcopy(nb)
    cpu.cells[0].source = """# تدريب كاشف الوجوه على CPU — بلا GPU
هذه نسخة لتدريب صور الوجوه الفعلية، وليست اختبار الأشكال. الافتراضي MODE=REAL_CPU وSOURCE=kaggle وUSE_DRIVE=False لتجنب مصادقة Drive تلقائيًا. يمكنك اختيار ZIP/Drive في المرحلة3، وتفعيل Drive للحفظ إن رغبت. لا تحتاج GPU أو ZIP المشروع، لكن تحتاج بيانات الوجوه وتصاريحها والإنترنت لتنزيل المتطلبات.

تُفحص البيانات وتقسم أولًا، ثم تُختار عينة حتى1000 صورة لكل فئة. تُستخرج ميزات MobileNetV3-Small مرة واحدة للتدريب والتحقق، ويُدرّب الرأس عليها مع جدولة التعلم وحفظ الأفضل والاستكمال. الاختبار مستقل، وتجربة JPEG تأتي بعد تثبيت النموذج والحدود.

لا نَعِد بدقة معينة؛ لم يُنفّذ تدريب مجموعة الوجوه في بيئتنا بسبب حجب Kaggle. اختبرنا سلامة التخزين والحساب والاستكمال برمجيًا؛ هذا ليس قياس دقة وجوه. تدريب الرأس على عينة محدودة قد لا يعمم على مولّدات أخرى. راقب F1/AUC ومعدل الاتهام الخاطئ والتغطية بعد تجربتك.

شغّل المراحل بالترتيب أو Run all بعد ضبط البيانات. إذا لم تُفعّل Drive نزّل النتائج قبل انتهاء الجلسة. التحميل والتدقيق قد يستغرقان معظم وقت أول تشغيل؛ لكل مرحلة عداد منفصل."""
    config_cell = [c for c in cpu.cells if c.cell_type == "code"][2]
    config_cell.source = config_cell.source.replace('MODE = "QUICK"', 'MODE = "REAL_CPU"', 1)
    cpu.metadata["colab"]["name"] = "FaceJPEG_CPU_Training.ipynb"
    nbf.validate(cpu)
    nbf.write(cpu, root / "FaceJPEG_CPU_Training.ipynb")
    predict = nbf.v4.new_notebook()
    predict.metadata = copy.deepcopy(nb.metadata)
    predict.metadata["colab"]["name"] = "FaceJPEG_Predict_Only.ipynb"
    predict_bootstrap = bootstrap.replace(
        'print("لا توجد CUDA متاحة؛ يمكن تدريب الوجوه بوضع REAL_CPU. QUICK اختبار أشكال فقط، وREAL يحتاج GPU.")',
        'print("التنبؤ متاح على CPU؛ لا يحتاج GPU أو بيانات التدريب.")',
    )
    install = [c.source for c in cells if c.cell_type == "code"][1].replace(
        '"requirements-colab.txt" if _torch_pair_ok', '"requirements-predict-colab.txt" if _torch_pair_ok',
    ).replace(
        'print(json.dumps(hardware_info(), ensure_ascii=False, indent=2))',
        'print("جاهز لتحميل best.pt؛ التنبؤ لا يحتاج GPU.")',
    )
    predict.cells = [
        nbf.v4.new_markdown_cell("""# FaceJPEG — اختبار صور جديدة بنموذج محفوظ
هذا دفتر للتنبؤ فقط من أربع خلايا؛ لا ينزل بيانات الوجوه ولا يعيد التدريب. يعمل على CPU.

**قبل مغادرة جلسة التدريب:** احفظ `best.pt` الناتج عن تدريب صور الوجوه على جهازك أو Drive. في دفتر التدريب يظهر مساره في المرحلة9، ويمكن تنزيله من لوحة Files أو استخراج ملفه من ZIP النتائج. حفظ الدفتر وحده لا يحفظ الأوزان. `last.pt` للاستكمال وليس بديلًا عن `best.pt` المختار بحدود التحقق.

شغّل الخلايا1–4 بالترتيب، وارفع ملف النموذج في الخلية3، ثم الصور من الواجهة في الخلية4. لا تحتاج حساب Kaggle أو تركيب Drive؛ يمكنك رفع النموذج من جهازك. أوزان QUICK الهندسية مرفوضة هنا لأنها لا تكشف أصالة الوجوه. لا توجد أوزان كشف وجوه حقيقية مضمّنة.

إغلاق تبويب Colab قد يُبقي الجلسة مدة، لكن الجلسة ورابط Gradio مؤقتان. إذا انتهت الجلسة أعد هذه الخلايا بملف النموذج المحفوظ، وافتح رابط الواجهة الجديد. ما دامت الواجهة تعمل اختبر صورًا متتابعة دون تشغيل الخلايا مجددًا. الدرجة ترجيح غير معاير وليست إثبات أصالة."""),
        nbf.v4.new_markdown_cell("## 1 — تهيئة الكود\nيفك الكود المضمن ويعرّف IN_COLAB والمسارات؛ شغّله أولًا في كل جلسة جديدة."),
        nbf.v4.new_code_cell(predict_bootstrap.strip()),
        nbf.v4.new_markdown_cell("## 2 — تثبيت متطلبات التنبؤ\nيحافظ على زوج Torch/Torchvision في Colab ويثبت متطلبات الواجهة فقط. لا ينزل مجموعة صور أو أوزان ImageNet."),
        nbf.v4.new_code_cell(install),
        nbf.v4.new_markdown_cell("""## 3 — تحميل النموذج المحفوظ
في Colab اترك CHECKPOINT_PATH فارغًا واختر ملف `best.pt` من جهازك عند ظهور زر الرفع. أو اكتب مسار ملف موجود في الجلسة. محليًا اكتب مساره. يُفحص ترتيب الفئات والمعالجة وحدود القرار المحفوظة. لا تغيّر الحدود ولا تحتاج بيانات التدريب/الاختبار للتنبؤ."""),
        nbf.v4.new_code_cell('''
MODEL_READY = False
PREDICT_CHECKPOINT = None
CHECKPOINT_PATH = ""  # مثال /content/best.pt؛ اتركه فارغًا للرفع من جهازك
from facejpeg.infer import Predictor
if not CHECKPOINT_PATH.strip():
    if not IN_COLAB:
        raise ValueError("اكتب مسار best.pt في CHECKPOINT_PATH ثم أعد هذه الخلية.")
    from google.colab import files
    _uploaded = files.upload()
    if len(_uploaded) != 1:
        raise ValueError("اختر ملف نموذج واحدًا: best.pt الناتج عن تدريب صور الوجوه.")
    CHECKPOINT_PATH = next(iter(_uploaded))
if not Path(CHECKPOINT_PATH).is_file():
    raise FileNotFoundError("لم أجد ملف النموذج. ارفعه أو صحّح CHECKPOINT_PATH.")
_checked_predictor = Predictor(CHECKPOINT_PATH, device="cpu")
if not _checked_predictor.face_classification_enabled:
    del _checked_predictor
    raise ValueError("هذا نموذج اختبار أشكال QUICK؛ لا يكشف أصالة الوجوه. اختر best.pt بعد تدريب REAL_CPU أو REAL على بيانات الوجوه. تغيير الاسم أو MODE لا يكفي.")
PREDICT_CHECKPOINT = Path(CHECKPOINT_PATH).resolve()
MODEL_READY = True
print("تم تحميل نموذج تجربة الوجوه:", PREDICT_CHECKPOINT.name)
print("حدود القرار المحفوظة:", _checked_predictor.bundle["thresholds"])
del _checked_predictor
'''.strip()),
        nbf.v4.new_markdown_cell("""## 4 — فتح الواجهة واختبار الصور
ارفع JPG/PNG لوجه واحد، واختر جودة JPEG ثم اضغط التحليل. لتجربة صورة أخرى بدّل الصورة واضغط الزر؛ لا تعِد التدريب أو تحميل النموذج. في Colab يظهر رابط عام مؤقت؛ لا ترفع صورًا حساسة. إذا انتهت الجلسة افتح هذا الدفتر مجددًا وشغّل1–4 بملف best.pt نفسه."""),
        nbf.v4.new_code_cell('''
if not globals().get("MODEL_READY", False) or not globals().get("PREDICT_CHECKPOINT"):
    raise RuntimeError("شغّل الخلايا1–3 وحمّل best.pt الصالح أولًا.")
from app import create_app
if globals().get("demo") is not None:
    demo.close()
demo = create_app(PREDICT_CHECKPOINT)
demo.launch(share=IN_COLAB, debug=False)
'''.strip()),
    ]
    nbf.validate(predict)
    nbf.write(predict, root / "FaceJPEG_Predict_Only.ipynb")
    build_jpeg_notebook(root, raw_notebook)
    print(f"Final notebook and compatibility copies: {(root/'FaceJPEG_Final_Colab.ipynb').stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
