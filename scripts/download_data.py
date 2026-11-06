"""Public Kaggle download; fallback ZIP/Drive is documented and available in Colab."""
import argparse
import json
from pathlib import Path


DATASET = "xhlulu/140k-real-and-fake-faces"


def download(output):
    import kagglehub
    # Never collect keys in code/chat. Kaggle may require account/acceptance in its own UI.
    try:
        root = Path(kagglehub.dataset_download(DATASET))
    except Exception as exc:
        raise RuntimeError("تعذّر تنزيل Kaggle. في الخلية 2 اختر SOURCE='zip' أو SOURCE='drive' ثم أعد المرحلة 4؛ لا تضع مفتاحك في الدفتر.") from exc
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"dataset": DATASET, "root": str(root)}, indent=2), encoding="utf-8")
    print(f"RAW_ROOT = {str(root)!r}")
    return root


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--record", default="data/download.json")
    a = p.parse_args()
    download(a.record)
