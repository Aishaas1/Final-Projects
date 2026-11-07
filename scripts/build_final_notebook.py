"""One self-contained, versioned notebook; reproducible source, no project ZIP prompt."""
import base64
import gzip
import hashlib
import json
import copy
from pathlib import Path

import nbformat as nbf


def build():
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in ["app.py", "requirements.txt", "requirements-colab.txt"]]
    paths += sorted((root / "facejpeg").glob("*.py"))
    paths += [root / "scripts" / name for name in ["make_demo.py", "download_data.py"]]
    paths += [root / "docs" / name for name in ["DATASET_CARD_AR.md", "PRESENTATION_AR.md"]]
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
## دفتر نهائي مستقل، الإصدار 2.2
الكود مضمن هنا؛ **لا تحتاج ZIP المشروع**. QUICK اختبار أشكال هندسية وواجهة JPEG بلا أحكام أصالة. لتدريب الوجوه اختر REAL على GPU أو REAL_CPU بلا GPU، ثم وفّر البيانات ودرّب النموذج. تغيير MODE وحده لا يكفي، ولا توجد أوزان كشف وجوه حقيقية مضمّنة.

في أول تجربة شغّل Runtime → Run all بالإعدادات الافتراضية. لتدريب الوجوه بلا GPU اختر REAL_CPU؛ لوضع REAL فعّل GPU. حدّد البيانات واسم تجربة جديدًا في المرحلة3. ZIP البيانات يُرفع من لوحة Files، وDrive يحتاج موافقتك.

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
PROJECT = (_base / "facejpeg_final_v2_2").resolve()
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
MODE = "QUICK"  # @param ["QUICK", "REAL", "REAL_CPU"]
SOURCE = "kaggle"  # @param ["kaggle", "zip", "drive"]
EXPERIMENT = "quick_v2_2" if MODE == "QUICK" else "faces_cpu_v2_2" if MODE == "REAL_CPU" else "faces_gpu_v2_2"
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
    if "demo" in globals():
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
    nbf.validate(nb)
    for name in ["FaceJPEG_Final_Colab.ipynb", "FaceJPEG_Standalone.ipynb", "FaceJPEG_Colab.ipynb"]:
        nbf.write(nb, root / name)
    cpu = copy.deepcopy(nb)
    cpu.cells[0].source = """# تدريب كاشف الوجوه على CPU — بلا GPU
هذه نسخة لتدريب صور الوجوه الفعلية، وليست اختبار الأشكال. الافتراضي MODE=REAL_CPU وSOURCE=kaggle وUSE_DRIVE=True لحفظ التقدم. يمكنك اختيار ZIP/Drive في المرحلة3. لا تحتاج GPU أو ZIP المشروع، لكن تحتاج بيانات الوجوه وتصاريحها والإنترنت لتنزيل المتطلبات.

تُفحص البيانات وتقسم أولًا، ثم تُختار عينة حتى1000 صورة لكل فئة. تُستخرج ميزات MobileNetV3-Small مرة واحدة للتدريب والتحقق، ويُدرّب الرأس عليها مع جدولة التعلم وحفظ الأفضل والاستكمال. الاختبار مستقل، وتجربة JPEG تأتي بعد تثبيت النموذج والحدود.

لا نَعِد بدقة معينة؛ لم يُنفّذ تدريب مجموعة الوجوه في بيئتنا بسبب حجب Kaggle. اختبرنا سلامة التخزين والحساب والاستكمال برمجيًا؛ هذا ليس قياس دقة وجوه. تدريب الرأس على عينة محدودة قد لا يعمم على مولّدات أخرى. راقب F1/AUC ومعدل الاتهام الخاطئ والتغطية بعد تجربتك.

شغّل المراحل بالترتيب أو Run all بعد ضبط البيانات والسماح بتركيب Drive. التحميل والتدقيق قد يستغرقان معظم وقت أول تشغيل؛ لكل مرحلة عداد منفصل."""
    config_cell = [c for c in cpu.cells if c.cell_type == "code"][2]
    config_cell.source = config_cell.source.replace('MODE = "QUICK"', 'MODE = "REAL_CPU"', 1)
    config_cell.source = config_cell.source.replace('USE_DRIVE = False', 'USE_DRIVE = True', 1)
    cpu.metadata["colab"]["name"] = "FaceJPEG_CPU_Training.ipynb"
    nbf.validate(cpu)
    nbf.write(cpu, root / "FaceJPEG_CPU_Training.ipynb")
    print(f"Final notebook and compatibility copies: {(root/'FaceJPEG_Final_Colab.ipynb').stat().st_size:,} bytes")


if __name__ == "__main__":
    build()
