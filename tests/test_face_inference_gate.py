import ast
from pathlib import Path
import sys
import types
from unittest.mock import Mock

import nbformat
import pytest
from PIL import Image

from facejpeg.common import PREPROCESS
from facejpeg.infer import Predictor


@pytest.mark.parametrize("filename", ["real_nasa.png", "ai_generated.png"])
def test_demo_never_scores_either_known_provenance_face(filename):
    predictor = Predictor("artifacts/review/quick/best.pt", "cpu")
    predictor.score = Mock(side_effect=AssertionError("A shapes checkpoint must not classify a human face"))
    result = predictor.compare_details(Path("examples/test_images") / filename, 30)
    predictor.score.assert_not_called()
    assert result[0].size == result[1].size == (224, 224)
    assert result[2] == result[3] == "كشف الأصالة غير متاح — النموذج غير مدرّب على صور الوجوه."
    assert result[5:] == (None, None)
    assert "يرجّح أنها" not in " ".join(result[2:5])


@pytest.mark.parametrize("score,expected", [(.1, "يرجّح أنها صورة فوتوغرافية"),
                                            (.5, "النتيجة غير حاسمة"),
                                            (.9, "يرجّح أنها مولَّدة بالذكاء الاصطناعي")])
def test_research_branch_keeps_threshold_based_decisions(score, expected, monkeypatch):
    # Pure unit stub, not a fake research checkpoint or claim of trained accuracy.
    predictor = Predictor.__new__(Predictor)
    predictor.bundle = {"data_kind": "research_faces", "run_label": "REAL_TRAINING", "quick": False,
                        "preprocess": PREPROCESS, "thresholds": {"low": .425, "high": .55}}
    predictor.score = Mock(return_value=score)
    monkeypatch.setattr("facejpeg.infer.validated_image", lambda _: Image.new("RGB", (224, 224)))
    result = predictor.compare_details("unused", 70)
    assert predictor.score.call_count == 2
    assert result[2].startswith(expected) and result[3].startswith(expected)
    assert result[5]["AI-Generated — مولّدة"] == score


def test_demo_score_components_hidden_and_no_top_class_label():
    from app import create_app
    app = create_app("artifacts/review/quick/best.pt")
    tables = [c for c in app.config["components"] if c["type"] == "dataframe"]
    assert len(tables) == 2 and all(c["props"]["visible"] is False for c in tables)
    assert not any(c["type"] == "label" for c in app.config["components"])
    app.close()


def test_new_notebook_purges_old_project_modules_but_keeps_current_ones(tmp_path):
    nb = nbformat.read("FaceJPEG_Final_Colab.ipynb", as_version=4)
    first = next(c.source for c in nb.cells if c.cell_type == "code")
    fn = next(n for n in ast.parse(first).body if isinstance(n, ast.FunctionDef) and n.name == "_refresh_project_imports")
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {"sys": sys, "Path": Path, "PROJECT": tmp_path}
    exec(compile(module, "refresh_imports", "exec"), scope)
    before = dict(sys.modules)
    old, current = types.ModuleType("facejpeg.old_test"), types.ModuleType("facejpeg.current_test")
    old.__file__ = "/tmp/old-facejpeg-project/infer.py"
    current.__file__ = str(tmp_path / "infer.py")
    try:
        sys.modules[old.__name__], sys.modules[current.__name__] = old, current
        removed = scope["_refresh_project_imports"]()
        assert old.__name__ in removed
        assert current.__name__ in sys.modules and sys.modules["torch"] is before["torch"]
    finally:
        for name in list(sys.modules):
            if name not in before:
                sys.modules.pop(name, None)
        sys.modules.update(before)
