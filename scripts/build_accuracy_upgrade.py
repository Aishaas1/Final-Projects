"""Build one self-contained refinement cell to run inside the working notebook."""
import base64
import gzip
import hashlib
import json
from pathlib import Path

import nbformat


CELL = '''# شغّل هذه الخلية داخل دفتر FaceJPEG الذي أكمل التدريب؛ لا يعاد تنزيل البيانات.
ACCURACY_MAX_EPOCHS = 8
ACCURACY_EVALUATE_FROZEN_MODEL = True
import base64, gzip, hashlib, json, sys, zipfile
from pathlib import Path

def _run_facejpeg_accuracy_upgrade():
    _state = globals()
    _prepared_value = _state.get("PREPARED")
    if not _prepared_value or not (Path(_prepared_value) / "manifest.csv").is_file():
        print("شغّلي هذه الخلية داخل دفتر FaceJPEG الذي أكمل التدريب، وفي جلسة ما زالت تحتوي PREPARED وbest.pt. إذا انتهت الجلسة، استعيدي مجلد prepared والأوزان من النسخة المحفوظة أولًا.")
        return
    _prepared = Path(_prepared_value).resolve()
    if _state.get("ACCURACY_ORIGINAL_PREPARED") == str(_prepared):
        _sources = dict(_state.get("ACCURACY_ORIGINAL_CHECKPOINTS", {}))
        _incumbent = _state.get("ACCURACY_ORIGINAL_INCUMBENT")
    else:
        _sources = dict(_state.get("CHECKPOINTS", {}))
        if not _sources and _state.get("BEST"):
            _sources = {"current": _state["BEST"]}
        _current = Path(_state["BEST"]).resolve() if _state.get("BEST") else None
        _incumbent = next((str(name) for name, path in _sources.items() if path and Path(path).resolve() == _current), None)
    _sources = {str(name): Path(path).resolve() for name, path in _sources.items()
                if path and Path(path).is_file()}
    if not _sources:
        print("لم أجد أوزان التدريب الحالي؛ أكمل المرحلة6 أو استعد ملفات best.pt أولًا. لن نستخدم أوزان QUICK بديلة.")
        return
    _blob = base64.b64decode(__BLOB__)
    assert hashlib.sha256(_blob).hexdigest() == __DIGEST__, "Upgrade source checksum failed"
    _files = json.loads(gzip.decompress(_blob).decode("utf-8"))
    _base = Path(_state.get("PROJECT", _prepared.parent)).resolve().parent
    _workspace = _base / ("facejpeg_accuracy_" + __DIGEST__[:12])
    for _relative, _content in _files.items():
        _path = (_workspace / _relative).resolve()
        if not _path.is_relative_to(_workspace.resolve()):
            raise ValueError("Unsafe upgrade source path")
        if _path.exists() and _path.read_text(encoding="utf-8") != _content:
            raise FileExistsError("مجلد التحسين يحتوي تعديلات محلية؛ احتفظ بها واستخدم مجلدًا جديدًا.")
        _path.parent.mkdir(parents=True, exist_ok=True)
        _path.write_text(_content, encoding="utf-8")
    sys.path.insert(0, str(_workspace))
    # Preserve Torch/Torchvision and the running UI's existing objects.
    for _name in list(sys.modules):
        if _name == "facejpeg" or _name.startswith("facejpeg.") or _name == "app":
            sys.modules.pop(_name, None)
    import torch
    import pandas as pd
    from facejpeg.common import file_sha256
    from facejpeg.infer import Predictor
    from facejpeg.upgrade import improve_accuracy
    from IPython.display import display
    _kind = json.loads((_prepared / "audit.json").read_text())["data_kind"]
    if _kind != "research_faces":
        print("هذه بيانات اختبار أشكال QUICK، وليست وجوهًا. يلزم إكمال تدريب الوجوه قبل تحسينه؛ لم نغير درجات الواجهة.")
        return
    _gpu = bool(torch.cuda.is_available())
    print("تحسين من الأوزان الحالية —", "GPU" if _gpu else "CPU: ضبط آخر كتلتين قد يستغرق وقتًا أطول من تدريب الرأس")
    _key_data = {"code": __DIGEST__, "manifest": file_sha256(_prepared / "manifest.csv"),
                 "sources": {name: file_sha256(path) for name, path in _sources.items()},
                 "epochs": ACCURACY_MAX_EPOCHS, "gpu": _gpu, "incumbent": _incumbent}
    _key = hashlib.sha256(json.dumps(_key_data, sort_keys=True).encode()).hexdigest()[:10]
    _old_run = Path(_state.get("RUN", _prepared.parent)).resolve()
    _output = _old_run.parent / (_old_run.name + "_accuracy_" + _key)
    _best, _selection = improve_accuracy(_prepared, _sources, _output, max_epochs=ACCURACY_MAX_EPOCHS,
                                          incumbent_name=_incumbent)
    _checked = Predictor(_best, "cpu")
    if not _checked.face_classification_enabled:
        raise ValueError("النموذج المختار ليس أوزان وجوه؛ لم تُستبدل الواجهة.")
    del _checked
    _state["ACCURACY_ORIGINAL_PREPARED"] = str(_prepared)
    _state["ACCURACY_ORIGINAL_CHECKPOINTS"] = _sources
    _state["ACCURACY_ORIGINAL_INCUMBENT"] = _incumbent
    _state["ACCURACY_UPGRADE_RUN"] = _output
    _state["BEST"] = _best
    _state["TRAINING_READY"] = True
    _reports = _selection["reports"]
    display(pd.DataFrame.from_dict(_reports, orient="index"))
    _anchor = _selection["incumbent"]
    _chosen = _selection["selected"]
    _gain = 100 * (_reports[_chosen]["mean_balanced_accuracy"] - _reports[_anchor]["mean_balanced_accuracy"])
    if _selection["improved_on_validation"]:
        print(f"اختير {_chosen}: تحسن متوسط الدقة المتوازنة على التحقق بمقدار {_gain:.2f} نقطة مئوية، دون خفض نتيجته على الأصل. هذه ليست دقة اختبار نهائي.")
    else:
        print("لم يتفوق المرشح الجديد في التحقق؛ احتفظنا بأفضل نموذج حالي بدل استبداله بنتيجة أضعف.")
    if ACCURACY_EVALUATE_FROZEN_MODEL:
        print("ثُبّت النموذج بالتحقق فقط. التقييم التالي متابعة على الاختبار السابق، ولا يدخل في الاختيار.")
        from facejpeg.evaluate import evaluate
        try:
            _test_table = evaluate(_prepared, _best, _output / "followup_test", device="cuda" if _gpu else "cpu")
            display(_test_table[["quality", "n_total", "binary_accuracy_at_0_5_all", "roc_auc_all",
                                 "fake_detected", "n_fake", "real_falsely_accused", "n_real", "abstain_total"]])
        except (OSError, ValueError, RuntimeError) as _test_error:
            print("النموذج المختار محفوظ، لكن تقييم المتابعة لم يكتمل:", str(_test_error))
    _zip_path = _output / "selected_model_results.zip"
    with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_DEFLATED) as _zip:
        for _name in ["best.pt", "selection.json", "upgrade_plan.json", "selected_protocol.json", "validation_comparison.csv"]:
            _zip.write(_output / _name, _name)
        for _name in ["jpeg_metrics.csv", "REPORT_AR.md", "evaluation_protocol.json", "predictions.csv",
                      "jpeg_effect.png", "confusion_matrices.png", "roc_curves.png"]:
            _path = _output / "followup_test" / _name
            if _path.is_file():
                _zip.write(_path, "followup_test/" + _name)
    print("النموذج المختار:", _best)
    print("نزّلي ZIP الأفضل ونتائج التحقق من لوحة Files قبل انتهاء الجلسة:", _zip_path)
    print("مجلد التحسين يحفظ أيضًا أوزان المرشحين والاستكمال؛ ZIP الصغير يضم النموذج المختار فقط.")
    # Reopen the same UI only after the new selection is ready; never change RUN or CHECKPOINTS.
    if _state.get("demo") is not None and _state.get("LAUNCH_UI", True):
        from app import create_app
        _new_demo = create_app(_best)
        _state["demo"].close()
        _state["demo"] = _new_demo
        _new_demo.launch(share=bool(_state.get("IN_COLAB", False) and _state.get("SHARE_GRADIO", True)), debug=False)
    else:
        print("شغّلي خلية الواجهة10 فقط لتحميل BEST المختار. واجهة الرفع نفسها لم تتغير.")

_run_facejpeg_accuracy_upgrade()
'''


def build():
    root = Path(__file__).resolve().parents[1]
    files = [root / "app.py"] + sorted((root / "facejpeg").glob("*.py"))
    source = json.dumps({p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in files},
                        ensure_ascii=False).encode()
    blob = gzip.compress(source, mtime=0)
    cell = CELL.replace("__BLOB__", repr(base64.b64encode(blob).decode())).replace(
        "__DIGEST__", repr(hashlib.sha256(blob).hexdigest()))
    compile(cell, "accuracy_upgrade_cell.py", "exec")
    (root / "accuracy_upgrade_cell.py").write_text(cell, encoding="utf-8")
    book = nbformat.v4.new_notebook()
    book.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "colab": {"name": "FaceJPEG_Accuracy_Upgrade.ipynb", "provenance": []}}
    book.cells = [nbformat.v4.new_markdown_cell("""# تحسين النموذج داخل الجلسة الحالية
انسخ الخلية التالية إلى نهاية دفتر FaceJPEG الذي أكمل التدريب، وشغّلها في **الجلسة نفسها** التي تحتوي PREPARED وCHECKPOINTS أو BEST. فتح هذا الدفتر في جلسة جديدة لا ينقل ملفات الجلسة القديمة تلقائيًا.

تُحفظ الأوزان الأصلية ولا يعاد تنزيل البيانات أو تقسيمها. تبدأ المرشحات من الأوزان الحالية: ضبط الرأس، ثم آخر كتلتين على CPU أو أربع على GPU تلقائيًا، بمعدل تعلم أقل للمستخرج. يختار التحسين من التحقق فقط على الأصل وجودات90/70/50/30، ويحتفظ بالنموذج السابق ما لم يتحسن المتوسط دون انخفاض التحقق الأصلي. لا يعرض درجات أقوى اصطناعيًا ولا يغير الواجهة أو حدودها يدويًا، ولا يضمن تحسن الاختبار.

بعد تثبيت الاختيار فقط، يحسب تقييم متابعة على الاختبار السابق؛ لا تُستخدم نتائجه لاختيار المرشح. إذا كانت الواجهة مفتوحة تُحدّث بالأوزان المختارة ويظهر رابط جديد بالشكل نفسه. احفظ selected_model_results.zip قبل انتهاء الجلسة. تظل RUN وPREPARED وCHECKPOINTS الأصلية دون تغيير؛ BEST يصبح الملف المختار.

الصور والبيانات والأوزان لا تُرسل إلى مستودع أو خدمة خارجية بواسطة هذه الخلية. تحتاج بيانات وأوزان وجوه فعلية؛ ترفض QUICK. التدريب الإضافي أبطأ من الاستدلال، خاصة على CPU؛ راقب عدادات التدريب ولا تفترض دقة أو زمنًا قبل القياس."""),
                  nbformat.v4.new_code_cell(cell)]
    nbformat.validate(book)
    nbformat.write(book, root / "FaceJPEG_Accuracy_Upgrade.ipynb")
    print("Built accuracy_upgrade_cell.py and FaceJPEG_Accuracy_Upgrade.ipynb")


if __name__ == "__main__":
    build()
