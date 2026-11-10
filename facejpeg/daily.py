"""Model-only restoration for fresh inference sessions; no training/data download."""
import json
from pathlib import Path, PurePosixPath
import pickle
import shutil
import tempfile
import warnings
import zipfile

from .common import file_sha256, write_json
from .infer import Predictor

MAX_MODEL_BYTES = 64 * 1024**2
MAX_ARCHIVE_BYTES = 512 * 1024**2


def verify_checkpoint(path, require_faces=True):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("لم أجد ملف نموذج صالحًا أو حجمه يتجاوز64MiB؛ ارفع best.pt من نتائج المشروع.")
    try:
        predictor = Predictor(path, device="cpu")
    except (ValueError, OSError, RuntimeError, EOFError, KeyError, TypeError, AttributeError, pickle.UnpicklingError) as error:
        raise ValueError("ملف النموذج تالف أو غير متوافق أو لا يتضمن حدود القرار. استخدم best.pt من النتائج، وليس last.pt.") from error
    if require_faces and not predictor.face_classification_enabled:
        raise ValueError("هذه أوزان اختبار أشكال QUICK؛ لا تكشف أصالة الوجوه. ارفع best.pt من تدريب الوجوه الفعلي أو ZIP نتائجه.")
    bundle = predictor.bundle
    del predictor
    return bundle


def _model_member(archive, preference):
    entries = []
    for entry in archive.infolist():
        path = PurePosixPath(entry.filename)
        if path.name != "best.pt" or entry.is_dir():
            continue
        if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
            raise ValueError("ZIP يحتوي مسار نموذج غير آمن؛ استخرج best.pt على جهازك وارفعه وحده.")
        if entry.file_size > MAX_MODEL_BYTES or entry.flag_bits & 1:
            raise ValueError("ملف النموذج داخل ZIP كبير جدًا أو مشفر؛ ارفع best.pt مباشرة.")
        rank = (0 if len(path.parts) == 1 else 1 if preference in path.parts else 2, len(path.parts))
        entries.append((rank, entry))
    if not entries:
        raise ValueError("لم أجد best.pt داخل ZIP. المطلوب ZIP نتائج التدريب أو النسخة الاحتياطية للنموذج، وليس ZIP الصور أو الكود فقط.")
    entries.sort(key=lambda row: row[0])
    if len(entries) > 1 and entries[0][0] == entries[1][0]:
        raise ValueError("يوجد أكثر من نموذج بالاختيار نفسه؛ استخرج best.pt المطلوب من ZIP وارفعه وحده.")
    if len(entries) > 1 and entries[0][0][0] == 2:
        raise ValueError("لا يمكن تحديد النموذج المقصود داخل ZIP؛ ارفع best.pt المطلوب مباشرة.")
    return entries[0][1]


def import_checkpoint(source, destination, preference="jpeg_augmented", require_faces=True):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError("لم أجد ملف النموذج. ارفع best.pt أو ZIP نتائجه أولًا.")
    if preference not in {"jpeg_augmented", "baseline"}:
        raise ValueError("اختيار النموذج يجب أن يكون baseline أو jpeg_augmented")
    if source.suffix.lower() not in {".pt", ".pth", ".zip"}:
        raise ValueError("هذه الخطوة لرفع النموذج best.pt أو ZIP نتائجه. صورة JPG/PNG تُرفع داخل الواجهة بعد تشغيلها.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-import-", dir=destination.parent) as directory:
        candidate = source
        member_name = None
        if source.suffix.lower() == ".zip":
            if source.stat().st_size > MAX_ARCHIVE_BYTES:
                raise ValueError("ZIP أكبر من512MiB؛ استخرج best.pt وارفعه مباشرة.")
            try:
                with zipfile.ZipFile(source) as archive:
                    entry = _model_member(archive, preference)
                    member_name = entry.filename
                    candidate = Path(directory) / "best.pt"
                    # Only the selected model is copied; archive paths are never extracted.
                    with archive.open(entry) as input_file, candidate.open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
                    if candidate.stat().st_size != entry.file_size:
                        raise ValueError("ملف النموذج داخل ZIP غير مكتمل.")
                    if "backup_manifest.json" in archive.namelist():
                        info = archive.getinfo("backup_manifest.json")
                        if info.file_size > 65536:
                            raise ValueError("بيانات النسخة الاحتياطية غير صالحة.")
                        record = json.loads(archive.read(info))
                        if not isinstance(record, dict):
                            raise ValueError("بيانات النسخة الاحتياطية غير صالحة.")
                        if record.get("checkpoint_sha256") != file_sha256(candidate):
                            raise ValueError("بصمة النموذج لا تطابق النسخة الاحتياطية؛ ارفع النسخة السليمة.")
            except (zipfile.BadZipFile, EOFError, OSError, json.JSONDecodeError) as error:
                raise ValueError("تعذر قراءة ZIP؛ الملف تالف أو ليس ZIP نتائج سليمًا.") from error
        bundle = verify_checkpoint(candidate, require_faces=require_faces)
        digest = file_sha256(candidate)
        if candidate.resolve() != destination.resolve():
            temporary = destination.with_suffix(".pt.tmp")
            shutil.copyfile(candidate, temporary)
            temporary.replace(destination)
        write_json(destination.parent / "loaded_model.json", {
            "checkpoint_sha256": digest, "classes": bundle["classes"],
            "preprocess": bundle["preprocess"], "thresholds": bundle["thresholds"],
            "data_kind": bundle["data_kind"], "source_member": member_name,
            "processing": "exact checkpoint bytes retained; no training or threshold changes"})
    return destination


def backup_checkpoint(checkpoint, output, require_faces=True):
    checkpoint, output = Path(checkpoint), Path(output)
    bundle = verify_checkpoint(checkpoint, require_faces=require_faces)
    record = {"backup_schema": 1, "checkpoint_sha256": file_sha256(checkpoint),
              "classes": bundle["classes"], "preprocess": bundle["preprocess"],
              "thresholds": bundle["thresholds"], "data_kind": bundle["data_kind"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(checkpoint, "best.pt")
        archive.writestr("backup_manifest.json", json.dumps(record, ensure_ascii=False, indent=2))
        archive.writestr("README_AR.txt", "احتفظ بهذا ZIP خارج جلسة Colab. في يوم آخر افتح FaceJPEG_Daily_Predict.ipynb وشغّل خليته وارفع هذا الملف. لا تحتاج صور التدريب للتنبؤ.\n")
    temporary.replace(output)
    return output


def discover_checkpoints(base, require_faces=True):
    """Inspect only known run locations, never scan the raw dataset or mount Drive."""
    base = Path(base)
    found = []
    patterns = ["facejpeg_final_*/runs/*/best.pt", "facejpeg_final_*/runs/*/jpeg_augmented/best.pt",
                "facejpeg_final_*/runs/*/baseline/best.pt", "facejpeg_daily_*/models/best.pt"]
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path in found:
                continue
            try:
                verify_checkpoint(path, require_faces=require_faces)
            except ValueError:
                continue
            found.append(path)
    return found


def inference_app(checkpoint):
    from app import create_app
    # These are known Gradio-5 compatibility notices, not runtime exceptions.
    with warnings.catch_warnings():
        for message in ["Setting 'api_name=False' in event listeners.*",
                        "The 'css' parameter in the Blocks constructor.*",
                        "Passing a tuple to 'row_count'.*", "Passing a tuple to 'col_count'.*"]:
            warnings.filterwarnings("ignore", message=message, category=DeprecationWarning)
        return create_app(checkpoint, device="cpu")
