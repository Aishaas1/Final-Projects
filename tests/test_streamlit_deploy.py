"""Optional frontend tests; no face training or accuracy claims."""
import io
from pathlib import Path

from PIL import Image
import pytest

pytest.importorskip("streamlit", minversion="1.67.0")
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "deploy/streamlit/app.py"
DEMO = ROOT / "artifacts/review/quick/best.pt"


@pytest.mark.parametrize("kind", ["missing", "corrupt"])
def test_streamlit_invalid_checkpoint_never_offers_classification(tmp_path, monkeypatch, kind):
    path = tmp_path / "best.pt"
    if kind == "corrupt":
        path.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("FACEJPEG_CHECKPOINT", str(path))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception
    assert not app.button and not app.file_uploader and not app.dataframe
    assert app.warning if kind == "missing" else app.error


@pytest.mark.parametrize("name", ["real_nasa.png", "ai_generated.png"])
def test_streamlit_demo_previews_faces_without_scores_or_judgments(monkeypatch, name):
    monkeypatch.setenv("FACEJPEG_CHECKPOINT", str(DEMO))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    assert not app.exception and "أشكال هندسية" in app.warning[0].value
    assert app.button[0].label == "معاينة وضغط JPEG فقط"
    path = ROOT / "examples/test_images" / name
    app.file_uploader[0].set_value((name, path.read_bytes(), "image/png"))
    app.selectbox[0].select(30)
    app.button[0].click().run(timeout=20)
    assert not app.exception and not app.error and not app.dataframe
    texts = [v.value for v in app.markdown]
    assert sum(t == "كشف الأصالة غير متاح — النموذج غير مدرّب على صور الوجوه." for t in texts) == 2
    assert not any("يرجّح أنها" in text or "درجة ترجيح التوليد:" in text for text in texts)
    assert len(app.image) == 2


@pytest.mark.parametrize("kind,expected", [("corrupt", "تالف"), ("none", "لم أجد"), ("multiple", "عدة وجوه")])
def test_streamlit_upload_errors_are_arabic(monkeypatch, kind, expected):
    monkeypatch.setenv("FACEJPEG_CHECKPOINT", str(DEMO))
    app = AppTest.from_file(str(APP)).run(timeout=20)
    if kind == "corrupt":
        data, name, mime = b"not an image", "bad.jpg", "image/jpeg"
    else:
        image = Image.new("RGB", (224, 224), "white")
        if kind == "multiple":
            with Image.open(ROOT / "examples/test_images/real_nasa.png") as face:
                face = face.convert("RGB")
            image = Image.new("RGB", (face.width * 2, face.height))
            image.paste(face, (0, 0))
            image.paste(face, (face.width, 0))
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        data, name, mime = stream.getvalue(), "example.png", "image/png"
    app.file_uploader[0].set_value((name, data, mime))
    app.button[0].click().run(timeout=20)
    assert not app.exception and expected in app.error[0].value
    assert not app.dataframe
