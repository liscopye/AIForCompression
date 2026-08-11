#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLS = (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001)


@dataclass(frozen=True)
class Job:
    dataset: str
    sample_token: str
    model: str

    @property
    def job_id(self) -> str:
        sample = self.sample_token.replace("vars000-", "v").replace("_", "-")
        return f"{self.dataset}_{self.model.lower().replace('-', '')}_{sample}"


JOBS = (
    Job("era5_npy", "vars000-267_t000-015", "CAESAR-D"),
    Job("era5_npy", "vars000-267_t000-015", "CAESAR-V"),
    Job("lysozyme", "test_chunks000", "CAESAR-D"),
    Job("lysozyme", "test_chunks031", "CAESAR-D"),
    Job("lysozyme", "test_chunks000", "CAESAR-V"),
    Job("lysozyme", "test_chunks031", "CAESAR-V"),
    Job("tomo", "projection0000", "CAESAR-D"),
    Job("tomo", "projection0989", "CAESAR-D"),
    Job("tomo", "projection0000", "CAESAR-V"),
    Job("tomo", "projection0989", "CAESAR-V"),
    Job("turb_rot_npz", "var000_sec000", "CAESAR-D"),
    Job("turb_rot_npz", "var000_sec008", "CAESAR-D"),
    Job("turb_rot_npz", "var000_sec000", "CAESAR-V"),
    Job("turb_rot_npz", "var000_sec008", "CAESAR-V"),
    Job("nyx", "baryon_density", "CAESAR-D"),
    Job("nyx", "baryon_density", "CAESAR-V"),
    Job("hurricane", "precip_log10", "CAESAR-D"),
    Job("hurricane", "precip_log10", "CAESAR-V"),
    Job("e3sm_npz", "t000-015", "CAESAR-D"),
    Job("e3sm_npz", "t400-415", "CAESAR-D"),
    Job("e3sm_npz", "t000-015", "CAESAR-V"),
    Job("e3sm_npz", "t400-415", "CAESAR-V"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-batch CAESAR objective-v1 jobs on eight GPUs.")
    parser.add_argument("--input-root", type=Path, default=Path("unified_results/objective_v1"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("unified_results/objective_v1_caesar_bs64_shards"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads-per-worker", type=int, default=8)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    return parser.parse_args()


def cpu_affinity(gpu: int) -> str:
    if not 0 <= gpu < 8:
        raise ValueError(f"No CPU affinity mapping for GPU {gpu}")
    base = gpu * 8
    sibling = base + 64
    return f"{base}-{base + 7},{sibling}-{sibling + 7}"


def missing_controls(summary_path: Path, job: Job, batch_size: int) -> list[float]:
    if not summary_path.exists():
        return list(CONTROLS)
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    model_prefix = "caesar_d" if job.model == "CAESAR-D" else "caesar_v"
    complete = {
        float(row["control"])
        for row in rows
        if "error" not in row
        and str(row.get("model_id", "")).startswith(model_prefix)
        and job.sample_token in str(row.get("canonical_sample_id", ""))
        and int(row.get("caesar_inference_batch_size", -1)) == batch_size
    }
    return [value for value in CONTROLS if value not in complete]


def command_for(args: argparse.Namespace, gpu: int, job: Job, controls: list[float]) -> list[str]:
    output = args.output_root / job.job_id
    return [
        "taskset", "-c", cpu_affinity(gpu), sys.executable, "-u",
        str(PROJECT_ROOT / "scripts/run_objective_benchmark.py"),
        "--dataset", job.dataset,
        "--gpu", str(gpu),
        "--input-root", str(args.input_root),
        "--output-root", str(output),
        "--models", job.model,
        "--sample-id-contains", job.sample_token,
        "--caesar-eb", *(f"{value:g}" for value in controls),
        "--caesar-batch-size", str(args.batch_size),
    ]


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_root / "logs"
    log_dir.mkdir(exist_ok=True)
    pending: queue.Queue[Job] = queue.Queue()
    for job in JOBS:
        pending.put(job)

    stop = threading.Event()
    active: dict[int, subprocess.Popen] = {}
    lock = threading.Lock()
    failures: list[tuple[str, int]] = []

    def request_stop(_signum=None, _frame=None) -> None:
        stop.set()
        with lock:
            processes = list(active.values())
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    def worker(gpu: int) -> None:
        while not stop.is_set():
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            output = args.output_root / job.job_id / job.dataset / "summary.json"
            controls = missing_controls(output, job, args.batch_size)
            if not controls:
                print(f"[skip-complete] {job.job_id}", flush=True)
                pending.task_done()
                continue
            command = command_for(args, gpu, job, controls)
            env = os.environ.copy()
            thread_count = str(args.threads_per_worker)
            env.update({
                "OMP_NUM_THREADS": thread_count,
                "MKL_NUM_THREADS": thread_count,
                "OPENBLAS_NUM_THREADS": thread_count,
                "NUMEXPR_NUM_THREADS": thread_count,
            })
            log_path = log_dir / f"{job.job_id}.log"
            print(
                f"[start] gpu={gpu} job={job.job_id} controls={','.join(f'{x:g}' for x in controls)}",
                flush=True,
            )
            started = time.monotonic()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n# {' '.join(command)}\n")
                log.flush()
                process = subprocess.Popen(
                    command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                with lock:
                    active[gpu] = process
                return_code = process.wait()
                with lock:
                    active.pop(gpu, None)
            elapsed = time.monotonic() - started
            print(
                f"[finish] gpu={gpu} job={job.job_id} rc={return_code} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )
            if return_code != 0 and not stop.is_set():
                failures.append((job.job_id, return_code))
            pending.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in args.gpus]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    manifest = {
        "batch_size": args.batch_size,
        "threads_per_worker": args.threads_per_worker,
        "gpus": args.gpus,
        "jobs": [job.job_id for job in JOBS],
        "failures": failures,
        "stopped": stop.is_set(),
    }
    (args.output_root / "schedule_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 130 if stop.is_set() else (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
