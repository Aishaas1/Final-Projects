"""Hardware diagnostics and progress that remains visible during slow operations."""
import json
import logging
import os
import threading
import time
from pathlib import Path

import torch


def hardware_info():
    result = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
              "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count()}
    if result["cuda_available"]:
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_name=torch.cuda.get_device_name(), gpu_total_gib=total / 2**30,
                      gpu_free_gib=free / 2**30, gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30)
    else:
        result["guidance_ar"] = ("لا توجد CUDA متاحة. اختر REAL_CPU لتدريب الوجوه على المعالج، أو QUICK لاختبار البرمجيات فقط. "
            "لوضع REAL الذي يضبط الشبكة اختر Runtime → Change runtime type → GPU. "
            "نسخة torch ذات اللاحقة +cpu لا تستخدم CUDA. هذا التشخيص لا يثبت سبب انتظار جلسة سابقة.")
    return result


def choose_batch_size(quick=False):
    if quick or not torch.cuda.is_available():
        return 8
    free, _ = torch.cuda.mem_get_info()
    # Conservative starting values, NOT a measured capacity or speed guarantee.
    return 64 if free >= 10 * 2**30 else 32 if free >= 5 * 2**30 else 16


def loader_options(device, quick=False, workers=None):
    n = 0 if quick or str(device) == "cpu" else min(2, os.cpu_count() or 1)
    n = n if workers is None else max(0, int(workers))
    return {"num_workers": n, "pin_memory": str(device).startswith("cuda"),
            "persistent_workers": n > 0, **({"prefetch_factor": 2} if n else {})}


class Monitor:
    """Heartbeat is observational; it never cancels slow training or downloads."""
    def __init__(self, output, interval=10):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.interval, self.started = interval, time.monotonic()
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.state = {"status": "running", "stage": "initializing", "completed": 0,
                      "total": None, "last_success": "monitor started"}
        self.stage_started = self.last_advance = self.started
        self.last_write = 0.
        self.logger = logging.getLogger(f"facejpeg.{self.output.resolve()}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
        for handler in [logging.StreamHandler(), logging.FileHandler(self.output / "events.log", encoding="utf-8")]:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def __enter__(self):
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def stage(self, name, total=None):
        with self.lock:
            self.state.update(stage=name, completed=0, total=total, status="running")
            self.stage_started = self.last_advance = time.monotonic()
            self.event(f"بدء المرحلة: {name}; total={total}")

    def advance(self, count=1, message=None):
        with self.lock:
            self.state["completed"] += count
            self.last_advance = time.monotonic()
            if message:
                self.state["last_success"] = message
            if time.monotonic() - self.last_write >= 2:
                self._write()

    def event(self, message):
        with self.lock:
            self.state["last_success"] = message
            self.logger.info(message)
            self._write()

    def _write(self):
        now = time.monotonic()
        elapsed = now - self.stage_started
        n, total = self.state["completed"], self.state["total"]
        rate = n / elapsed if elapsed and n else None
        snapshot = {**self.state, "elapsed_stage_seconds": elapsed,
                    "elapsed_run_seconds": now - self.started,
                    "seconds_since_progress": now - self.last_advance,
                    "units_per_second": rate,
                    "eta_seconds": max(0., (total - n) / rate) if total is not None and rate else None}
        if torch.cuda.is_available():
            snapshot["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            snapshot["gpu_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
        temporary = self.output / "status.json.tmp"
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output / "status.json")
        self.last_write = now
        return snapshot

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                s = self._write()
                note = " — العداد لم يتقدم؛ افحص آخر عملية، لا يعني ذلك وحده فشل العملية" if s["seconds_since_progress"] >= 60 else ""
                self.logger.info("%s | %s/%s | elapsed=%.1fs | ETA=%s | last=%s%s",
                    s["stage"], s["completed"], s["total"], s["elapsed_stage_seconds"],
                    round(s["eta_seconds"], 1) if s["eta_seconds"] is not None else "not yet measurable",
                    s["last_success"], note)

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=1)
        with self.lock:
            self.state["status"] = "failed" if exc_type else "completed"
            if exc_type:
                # Do not copy arbitrary SDK exceptions/URLs that may contain credentials into logs.
                self.state["error_type"] = exc_type.__name__
                self.logger.error("توقفت المرحلة %s بسبب %s؛ راجع رسالة الخطأ في الخلية.", self.state["stage"], exc_type.__name__)
            else:
                self.logger.info("اكتملت العملية؛ elapsed=%.2fs", time.monotonic() - self.started)
            self._write()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
