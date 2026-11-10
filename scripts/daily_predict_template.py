# رفع النموذج أولًا؛ رفع الصورة يكون داخل الواجهة التي تظهر في النهاية.
import base64, gzip, hashlib, importlib, importlib.metadata, json, os, re, subprocess, sys
from pathlib import Path

CHECKPOINT_PATH = os.environ.get("FACEJPEG_CHECKPOINT", "")  # اتركه فارغًا لزر الرفع في Colab
MODEL_CHOICE = "jpeg_augmented"  # @param ["jpeg_augmented", "baseline"]
DOWNLOAD_BACKUP = True  # @param {type:"boolean"}
_DAILY_PAYLOAD = "__DAILY_PAYLOAD__"
_DAILY_DIGEST = "__DAILY_DIGEST__"


def _daily_dependencies():
    """Install only inference dependencies; preserve Colab's Torch/CUDA pair."""
    def version(distribution):
        try:
            raw = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None
        digits = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", raw)
        return tuple(int(part or 0) for part in digits.groups()) if digits else None

    torch_version = version("torch")
    if torch_version is None or not (2, 5, 0) <= torch_version < (3, 0, 0) or version("torchvision") is None:
        raise RuntimeError("تحتاج هذه النسخة زوج PyTorch/Torchvision المتوافق المثبت في Colab، وTorch من2.5 إلى أقل من3. افتح جلسة Colab حديثة؛ لم نستبدل Torch بعجلات CPU.")
    checks = [
        ("numpy", ["numpy"], (1, 26, 0), (3, 0, 0), "numpy>=1.26,<3"),
        ("PIL", ["Pillow"], (10, 4, 0), (13, 0, 0), "Pillow>=10.4,<13"),
        ("cv2", ["opencv-python-headless", "opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"],
         (4, 10, 0), (5, 0, 0), "opencv-python-headless>=4.10,<5"),
        ("gradio", ["gradio"], (5, 50, 0), (5, 50, 1), "gradio==5.50.0"),
    ]
    missing = []
    for module, distributions, low, high, requirement in checks:
        installed = [version(name) for name in distributions]
        if any(value is not None and low <= value < high for value in installed):
            continue
        if module in sys.modules:
            raise RuntimeError("مكتبة " + module + " محمّلة بإصدار مختلف داخل الجلسة. احفظ best.pt ثم افتح جلسة جديدة وشغّل هذا الدفتر وحده؛ لا نبدّل مكتبة محمّلة أثناء عملها.")
        missing.append(requirement)
    if missing:
        print("1/4 — تثبيت متطلبات التنبؤ فقط. انتظر حتى ينتهي pip؛ لا نحمّل بيانات التدريب.", flush=True)
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError("لم يكتمل تثبيت المكتبات. راجع اتصال Colab ثم أعد تشغيل الخلية؛ لم نغيّر ملف نموذجك.") from error
        importlib.invalidate_caches()
    else:
        print("1/4 — متطلبات التنبؤ متوفرة؛ لا تثبيت إضافي.", flush=True)
    try:
        import torch, torchvision, numpy, PIL, cv2, gradio
        # Importing both detects a broken binary pair before the model upload.
        print("PyTorch", torch.__version__, "| Torchvision", torchvision.__version__, "| التنبؤ على CPU", flush=True)
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise RuntimeError("تعذر تحميل المكتبات أو أن زوج Torch/Torchvision غير متوافق. افتح جلسة Colab جديدة وشغّل هذا الدفتر أولًا. تفاصيل السبب: " + str(error)) from error


def _daily_sources(base):
    blob = base64.b64decode(_DAILY_PAYLOAD)
    if hashlib.sha256(blob).hexdigest() != _DAILY_DIGEST:
        raise ValueError("محتوى الدفتر غير مكتمل. افتح النسخة الأصلية من GitHub مجددًا.")
    files = json.loads(gzip.decompress(blob).decode("utf-8"))
    workspace = (base / ("facejpeg_daily_" + _DAILY_DIGEST[:12])).resolve()
    for relative, content in files.items():
        path = (workspace / relative).resolve()
        if not path.is_relative_to(workspace):
            raise ValueError("مسار كود غير صالح داخل الدفتر.")
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError("كود مجلد التنبؤ عُدّل محليًا. احتفظ بتعديلاتك ثم استخدم FACEJPEG_DAILY_HOME لمجلد جديد.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    elif sys.path[0] != str(workspace):
        sys.path.remove(str(workspace))
        sys.path.insert(0, str(workspace))
    # Refresh only this project's imports; never unload Torch/NumPy or train again.
    for name in list(sys.modules):
        if name == "facejpeg" or name.startswith("facejpeg.") or name == "app":
            module_file = getattr(sys.modules[name], "__file__", "")
            if not module_file or not Path(module_file).resolve().is_relative_to(workspace):
                sys.modules.pop(name, None)
    importlib.invalidate_caches()
    return workspace


def _daily_main(require_faces=True):
    state = globals()
    try:
        try:
            from google.colab import files as colab_files
            in_colab = True
        except ImportError:
            in_colab = False
        base = Path(os.environ.get("FACEJPEG_DAILY_HOME", "/content" if in_colab else str(Path.cwd()))).resolve()
        base.mkdir(parents=True, exist_ok=True)
        _daily_dependencies()
        workspace = _daily_sources(base)
        from facejpeg.daily import backup_checkpoint, discover_checkpoints, import_checkpoint, inference_app, verify_checkpoint
        source = Path(CHECKPOINT_PATH).expanduser().resolve() if CHECKPOINT_PATH else None
        if source is None:
            candidates = []
            for previous in [state.get("BEST"), workspace / "models/best.pt"]:
                if not previous:
                    continue
                try:
                    verify_checkpoint(previous, require_faces=require_faces)
                except ValueError:
                    continue
                candidates = [Path(previous).resolve()]
                break
            if not candidates:
                candidates = discover_checkpoints(base, require_faces=require_faces)
            if len(candidates) == 1:
                source = candidates[0]
                print("2/4 — وجدت نموذجك في ملفات الجلسة:", source, flush=True)
            elif len(candidates) > 1:
                print("2/4 — توجد عدة نماذج محفوظة؛ اختر رقم النموذج الذي تريد استخدامه:", flush=True)
                for number, candidate in enumerate(candidates, 1):
                    print(number, candidate)
                choice = input("رقم النموذج: ").strip()
                if not choice.isdigit() or not 1 <= int(choice) <= len(candidates):
                    raise ValueError("لم تحدد رقم نموذج صالحًا. أعد تشغيل الخلية للاختيار مجددًا.")
                source = candidates[int(choice) - 1]
        if source is None:
            if not in_colab:
                raise ValueError("حدّد مسار best.pt أو ZIP نتائجه في FACEJPEG_CHECKPOINT ثم أعد التشغيل. مثال: FACEJPEG_CHECKPOINT=/path/best.pt python daily_predict.py")
            print("2/4 — ارفع ملفًا واحدًا: best.pt من تدريبك الفعلي، أو ZIP نتائجه/النسخة الاحتياطية. الصورة ترفعها لاحقًا داخل الواجهة.", flush=True)
            upload_directory = workspace / "uploads"
            upload_directory.mkdir(exist_ok=True)
            previous_directory = Path.cwd()
            try:
                os.chdir(upload_directory)
                uploaded = colab_files.upload()
            finally:
                os.chdir(previous_directory)
            if len(uploaded) != 1:
                raise ValueError("ارفع ملف نموذج واحدًا فقط. إذا ألغيت الرفع أعد تشغيل الخلية عندما يكون best.pt جاهزًا.")
            name, content = next(iter(uploaded.items()))
            source = upload_directory / Path(name).name
            source.write_bytes(content)
            del content, uploaded
        checkpoint = import_checkpoint(source, workspace / "models/best.pt", preference=MODEL_CHOICE, require_faces=require_faces)
        backup = backup_checkpoint(checkpoint, workspace / "facejpeg_model_backup.zip", require_faces=require_faces)
        state["BEST"] = checkpoint
        state["FACEJPEG_MODEL_BACKUP"] = backup
        print("3/4 — تم تحميل الأوزان وحدود القرار والمعالجة نفسها دون تدريب.\nنسخة النموذج الاحتياطية:", backup, flush=True)
        if in_colab and DOWNLOAD_BACKUP:
            try:
                colab_files.download(str(backup))
                print("طلبت تنزيل ZIP. تأكد من وصوله إلى جهازك واحتفظ به قبل انتهاء جلسة Colab.", flush=True)
            except Exception as error:
                print("تعذر بدء التنزيل التلقائي؛ نزّل الملف من لوحة Files في Colab:", backup, "|", str(error), flush=True)
        if os.environ.get("FACEJPEG_NO_UI") == "1":
            return {"checkpoint": checkpoint, "backup": backup, "workspace": workspace}
        new_demo = inference_app(checkpoint)
        if state.get("demo") is not None:
            try:
                state["demo"].close()
            except (AttributeError, RuntimeError):
                pass
        state["demo"] = new_demo
        print("4/4 — فتح واجهة رفع الصورة. انتظر ظهور الرابط الجديد، ثم ارفع JPG/PNG فيها. رابط Gradio مؤقت ويحتاج استمرار الجلسة.", flush=True)
        new_demo.launch(share=in_colab, debug=False, max_file_size="15mb")
        return {"checkpoint": checkpoint, "backup": backup, "workspace": workspace, "demo": new_demo}
    except (ValueError, OSError, RuntimeError, EOFError) as error:
        print("لم يكتمل تشغيل الواجهة:", str(error), flush=True)
        print("ملفات النموذج الموجودة لم تُحذف. إذا انتهت الجلسة السابقة، ارفع best.pt أو نسخة ZIP محفوظة خارج Colab.", flush=True)
        return None


_DAILY_RESULT = _daily_main()
