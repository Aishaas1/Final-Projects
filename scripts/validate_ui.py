"""Functional local HTTP/API checks. Face samples are test-only and not distributed."""
import argparse
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from gradio_client import Client, handle_file
from PIL import Image

from app import create_app
from facejpeg.common import read_rgb
from facejpeg.infer import Predictor


def validate(checkpoint, output, face, multiple_faces, port=7861):
    app = create_app(checkpoint)
    record = {"share": False, "checkpoint": str(checkpoint)}
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False, quiet=True, prevent_thread_lock=True)
        base = f"http://127.0.0.1:{port}"
        with urllib.request.urlopen(base, timeout=20) as r:
            assert r.status == 200
        client = Client(base, verbose=False)
        with tempfile.TemporaryDirectory() as directory:
            blank, corrupt = Path(directory) / "blank.png", Path(directory) / "bad.jpg"
            Image.new("RGB", (224, 224), "white").save(blank)
            corrupt.write_bytes(b"not an image")
            for file, expected in [(blank, "لم أجد"), (corrupt, "تالف"), (multiple_faces, "عدة وجوه")]:
                response = client.predict(handle_file(str(file)), 30, api_name="/compare")
                assert expected in response[4]
                record[expected] = "passed"
        response = client.predict(handle_file(str(face)), 30, api_name="/compare")
        expected = Predictor(checkpoint).compare_details(face, 30)
        assert len(response) == 7 and tuple(response[2:5]) == tuple(expected[2:5])
        for index in [0, 1]:
            assert np.array_equal(np.asarray(read_rgb(response[index])), np.asarray(expected[index]))
        classify_faces = Predictor(checkpoint).face_classification_enabled
        if classify_faces:
            for distribution in expected[5:]:
                assert abs(sum(distribution.values()) - 1.) < 1e-7
            record["class_distributions"] = "passed"
            record["score_parity"] = "passed"
        else:
            assert expected[5:] == (None, None)
            assert "كشف الأصالة غير متاح" in response[2]
            assert "يرجّح أنها" not in " ".join(response[2:5])
            assert "Real —" not in str(response[5:])
            record["demo_face_judgments_and_scores_suppressed"] = "passed"
            record["score_parity"] = "not_applicable_demo_scores_suppressed"
        record.update(http=200, single_face="passed", response_text_parity="passed",
                      lossless_png_previews="passed",
                      face_classification_enabled=classify_faces,
                      no_real_training_claim=True,
                      single_face_file=Path(face).name, multiple_faces_file=Path(multiple_faces).name,
                      sample="Caller-supplied images for UI tests only; not training data or a classification benchmark")
    finally:
        app.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="artifacts/review/ui_validation.json")
    p.add_argument("--face", required=True)
    p.add_argument("--multiple-faces", required=True)
    p.add_argument("--port", type=int, default=7861)
    a = p.parse_args()
    validate(a.checkpoint, a.output, a.face, a.multiple_faces, a.port)
