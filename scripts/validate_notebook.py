"""Execute all final notebook stages locally, with Gradio launch disabled.

Run from the extracted project root in its installed environment. This validates
the default CPU fixture workflow; it does NOT emulate Colab uploads or Drive.
"""
import argparse
import base64
import contextlib
import io
import json
import os
from pathlib import Path

import nbformat
import pandas as pd
import IPython.display as ipdisplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/review/notebook")
    args = parser.parse_args()
    project_root = Path.cwd().resolve()
    output = (project_root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = project_root / "FaceJPEG_Final_Colab.ipynb"
    notebook = nbformat.read(path, as_version=4)
    namespace, completed = {"__name__": "__notebook__"}, []
    current = None

    def capture_display(*objects, **kwargs):
        for obj in objects:
            if isinstance(obj, pd.DataFrame):
                data = {"text/plain": obj.to_string(), "text/html": obj.to_html()}
            elif isinstance(obj, ipdisplay.Markdown):
                data = {"text/markdown": obj.data, "text/plain": obj.data}
            elif isinstance(obj, ipdisplay.Image):
                data = {"image/png": base64.b64encode(Path(obj.filename).read_bytes()).decode()}
            else:
                data = {"text/plain": repr(obj)}
            current.outputs.append(nbformat.v4.new_output("display_data", data=data, metadata={}))

    original_display = ipdisplay.display
    ipdisplay.display = capture_display
    try:
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            current = cell
            current.outputs = []
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                exec(compile(cell.source, f"notebook_stage_{len(completed)+1}", "exec"), namespace)
            if stream.getvalue():
                current.outputs.insert(0, nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
            current.execution_count = len(completed) + 1
            completed.append(len(completed) + 1)
            print(f"Notebook stage {len(completed)} passed", flush=True)
    finally:
        ipdisplay.display = original_display
    nbformat.validate(notebook)
    nbformat.write(notebook, output / "executed_quick.ipynb")
    record = {"executed_locally": completed, "source": "fixture", "quick": True,
              "gradio_launch_in_notebook": "disabled; tested separately by local API",
              "colab_drive_and_browser_runtime": "not executed in hosted Colab here"}
    (output / "validation.json").write_text(json.dumps(record, indent=2))
    os.chdir(project_root)


if __name__ == "__main__":
    os.environ.setdefault("FACEJPEG_NO_UI", "1")
    main()
