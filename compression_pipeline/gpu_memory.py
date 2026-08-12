from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Any


def run_with_gpu_peak(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    **popen_kwargs: Any,
) -> tuple[subprocess.CompletedProcess, float | None]:
    """Run one external CUDA command and sample its incremental GPU peak."""
    import pynvml

    process_env = env or os.environ.copy()
    visible = process_env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    physical_index = int(visible) if visible.isdigit() else 0
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
    baseline = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    peak = baseline
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
            time.sleep(0.002)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        proc = subprocess.run(command, env=process_env, **popen_kwargs)
    finally:
        stop.set()
        thread.join()
        peak = max(peak, int(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
        pynvml.nvmlShutdown()
    return proc, max(0.0, (peak - baseline) / 1024**2)
