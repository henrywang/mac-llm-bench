#!/usr/bin/env python3
"""Benchmark harness: runs all tests against one or both runners, logs to results/raw.csv."""

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
import requests

RUNNERS: dict[str, dict[str, str]] = {
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3.6:35b-a3b-nvfp4",
        "api_key": "",
    },
    "omlx": {
        "base_url": "http://localhost:8000/v1",
        "model": "Qwen3.6-35B-A3B-4bit",
        "api_key": os.getenv("OMLX_API_KEY", ""),
    },
}

PROMPTS = Path(__file__).parent.parent / "prompts"
RESULTS = Path(__file__).parent.parent / "results"
METRICS_DIR = RESULTS


def raw_csv(runner: str) -> Path:
    return RESULTS / f"raw_{runner}.csv"

CSV_FIELDS = [
    "run_id", "runner", "test_id", "task", "rep",
    "load_time_s", "ttft_ms", "gen_tps_burst", "gen_tps_sustained", "total_time_s",
    "quality_score",
    "peak_cpu_pct", "avg_cpu_pct",
    "peak_gpu_pct", "avg_gpu_pct",
    "peak_mem_gb", "avg_mem_gb",
    "peak_cpu_temp_c", "avg_cpu_temp_c",
    "peak_gpu_temp_c", "avg_gpu_temp_c",
    "peak_power_w", "avg_power_w",
    "notes",
]


# ---------------------------------------------------------------------------
# Memory sampler (psutil, runs in background thread)
# ---------------------------------------------------------------------------

class MemorySampler:
    def __init__(self, interval: float = 1.0) -> None:
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._interval = interval
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, float]:
        self._stop.set()
        self._thread.join()
        if not self._samples:
            return {"peak_mem_gb": 0.0, "avg_mem_gb": 0.0}
        return {
            "peak_mem_gb": round(max(self._samples), 2),
            "avg_mem_gb": round(sum(self._samples) / len(self._samples), 2),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(psutil.virtual_memory().used / 1024 ** 3)
            time.sleep(self._interval)


# ---------------------------------------------------------------------------
# powermetrics helpers
# ---------------------------------------------------------------------------

def start_powermetrics(run_id: str) -> tuple[subprocess.Popen[bytes], Path]:
    METRICS_DIR.mkdir(exist_ok=True)
    out_path = METRICS_DIR / f"metrics_{run_id}.txt"
    proc = subprocess.Popen(
        ["sudo", "powermetrics", "--samplers", "gpu_power,cpu_power,thermal", "-i", "1000"],
        stdout=open(out_path, "wb"),
        stderr=None,  # let sudo password prompt reach the terminal
    )
    time.sleep(2)  # wait for first sample to land before the LLM call starts
    return proc, out_path


def stop_powermetrics(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def parse_powermetrics(path: Path) -> dict[str, float]:
    """Extract cpu/gpu util, power, and temperatures from a powermetrics output file."""
    cpu_utils: list[float] = []
    gpu_utils: list[float] = []
    cpu_powers: list[float] = []
    gpu_powers: list[float] = []
    cpu_temps: list[float] = []
    gpu_temps: list[float] = []

    text = path.read_text(errors="replace")
    for line in text.splitlines():
        l = line.strip()
        if "CPU Power:" in l:
            cpu_powers.append(_parse_mw(l))
        elif "GPU Power:" in l:
            gpu_powers.append(_parse_mw(l))
        # Apple Silicon: per-core active residency (e.g. "CPU 0 active residency: 14.54%")
        elif l.startswith("CPU ") and "active residency:" in l and "HW" not in l:
            cpu_utils.append(_parse_pct(l))
        # Apple Silicon: GPU active residency (e.g. "GPU HW active residency: 2.98%")
        elif "GPU HW active residency:" in l:
            gpu_utils.append(_parse_pct(l))
        # Intel fallbacks
        elif "CPU Utilization:" in l or "CPU usage:" in l:
            cpu_utils.append(_parse_pct(l))
        elif "GPU Activity:" in l:
            gpu_utils.append(_parse_pct(l))
        elif "CPU die temperature:" in l:
            cpu_temps.append(_parse_celsius(l))
        elif "GPU die temperature:" in l:
            gpu_temps.append(_parse_celsius(l))

    def stats(vals: list[float], key_peak: str, key_avg: str) -> dict[str, float]:
        if not vals:
            return {key_peak: 0.0, key_avg: 0.0}
        return {key_peak: round(max(vals), 1), key_avg: round(sum(vals) / len(vals), 1)}

    return {
        **stats(cpu_utils, "peak_cpu_pct", "avg_cpu_pct"),
        **stats(gpu_utils, "peak_gpu_pct", "avg_gpu_pct"),
        **stats([w / 1000 for w in cpu_powers], "peak_cpu_power_w", "avg_cpu_power_w"),
        **stats([w / 1000 for w in gpu_powers], "peak_gpu_power_w", "avg_gpu_power_w"),
        **stats(cpu_temps, "peak_cpu_temp_c", "avg_cpu_temp_c"),
        **stats(gpu_temps, "peak_gpu_temp_c", "avg_gpu_temp_c"),
        "peak_power_w": round(max(cpu_powers + gpu_powers, default=0) / 1000, 1),
        "avg_power_w": round(
            sum(cpu_powers + gpu_powers) / max(len(cpu_powers + gpu_powers), 1) / 1000, 1
        ),
    }


def _parse_mw(line: str) -> float:
    for part in line.split():
        try:
            return float(part.replace("mW", ""))
        except ValueError:
            continue
    return 0.0


def _parse_pct(line: str) -> float:
    for part in line.split():
        try:
            return float(part.replace("%", ""))
        except ValueError:
            continue
    return 0.0


def _parse_celsius(line: str) -> float:
    for part in line.split():
        try:
            return float(part.replace("C", "").replace("°", ""))
        except ValueError:
            continue
    return 0.0


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------

def chat(base_url: str, model: str, prompt: str, max_tokens: int = 512, api_key: str = "") -> dict[str, Any]:
    """Stream a chat completion, return timing and tok/s."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},  # request completion_tokens in final chunk
        "think": False,  # Ollama extension; ignored by other runners; /no_think covers the rest
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    t_start = time.monotonic()
    ttft_ms = 0.0
    delta_count = 0  # fallback: count non-empty delta.content chunks
    usage_tokens: int | None = None
    first = True

    with requests.post(url, json=payload, headers=headers, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode().removeprefix("data: ")
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            # prefer authoritative completion_tokens from final chunk
            usage = chunk.get("usage") or {}
            if usage.get("completion_tokens"):
                usage_tokens = int(usage["completion_tokens"])
            choices = chunk.get("choices") or []
            if choices:
                d = choices[0].get("delta", {})
                # check content first, then thinking fields (Ollama/oMLX thinking models)
                delta = d.get("content") or d.get("thinking") or d.get("reasoning_content") or ""
            else:
                delta = ""
            if delta:
                if first:
                    ttft_ms = (time.monotonic() - t_start) * 1000
                    first = False
                delta_count += 1

    total_s = time.monotonic() - t_start
    token_count = usage_tokens if usage_tokens is not None else delta_count
    # use total_s as denominator: gen_time_s ≈ 0 for thinking models (ttft ≈ total)
    # makes tps blow up, total_s gives stable end-to-end throughput
    tps = token_count / total_s if total_s > 0 else 0.0
    return {"ttft_ms": round(ttft_ms, 1), "total_time_s": round(total_s, 2), "tps": round(tps, 1)}


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def run_t1(runner: str, cfg: dict[str, str], rep: int) -> dict[str, Any]:
    """Cold-start: time from first request to first token. Runner must be freshly started."""
    result = chat(cfg["base_url"], cfg["model"], "Say hello in one sentence.", max_tokens=20, api_key=cfg["api_key"])
    return {
        "test_id": "t1", "task": "cold_start", "rep": rep,
        "load_time_s": round(result["ttft_ms"] / 1000, 2),
        "ttft_ms": result["ttft_ms"],
        "total_time_s": result["total_time_s"],
        "gen_tps_burst": "", "gen_tps_sustained": "", "quality_score": "",
    }


def run_t2(runner: str, cfg: dict[str, str], rep: int) -> dict[str, Any]:
    prompt = (PROMPTS / "t2_synthetic.txt").read_text()
    result = chat(cfg["base_url"], cfg["model"], prompt, api_key=cfg["api_key"], max_tokens=512)
    return {
        "test_id": "t2", "task": "raw_tps", "rep": rep,
        "load_time_s": "", "ttft_ms": result["ttft_ms"],
        "total_time_s": result["total_time_s"],
        "gen_tps_burst": result["tps"] if rep == 1 else "",
        "gen_tps_sustained": result["tps"] if rep == 3 else "",
        "quality_score": "",
    }


def run_t3(runner: str, cfg: dict[str, str], rep: int) -> dict[str, Any]:
    prompt = (PROMPTS / "t3_large_pr.txt").read_text()
    result = chat(cfg["base_url"], cfg["model"], prompt, api_key=cfg["api_key"], max_tokens=50)
    return {
        "test_id": "t3", "task": "ttft_prefill", "rep": rep,
        "load_time_s": "", "ttft_ms": result["ttft_ms"],
        "total_time_s": result["total_time_s"],
        "gen_tps_burst": "", "gen_tps_sustained": result["tps"],
        "quality_score": "",
    }


T4_PI_PROVIDER = {"ollama": "ollama", "omlx": "omlx"}

def run_t4(runner: str, cfg: dict[str, str], task: str, rep: int) -> dict[str, Any]:
    """Run a T4 coding task via the Pi agent and record wall-clock time."""
    prompt_file = {"task_a": "t4_task_a.txt", "task_b": "t4_task_b_broken.py"}[task]
    prompt = (PROMPTS / prompt_file).read_text()
    provider = T4_PI_PROVIDER[runner]
    model = cfg["model"]

    project_root = Path(__file__).parent.parent
    work_dir = project_root  # Pi must run here to load project-local extensions

    # snapshot files before Pi runs so we can move anything new afterwards
    before = set(work_dir.glob("*"))

    out_dir = RESULTS / f"t4_{runner}_{task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pi_output.txt"

    t_start = time.monotonic()
    with open(out_path, "w") as out_file:
        proc = subprocess.Popen(
            ["pi", "--approve", "--provider", provider, "--model", model,
             "--no-session", "--print", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=work_dir,
        )
        for line in proc.stdout:
            text = line.decode(errors="replace")
            print(text, end="", flush=True)
            out_file.write(text)
        proc.wait()
    total_s = time.monotonic() - t_start

    # move any new files Pi created in project root into the output folder
    new_files = set(work_dir.glob("*")) - before
    for f in new_files:
        if f.is_file():
            f.rename(out_dir / f.name)
            print(f"  Moved {f.name} → {out_dir.relative_to(project_root)}/")

    if proc.returncode != 0:
        print(f"  WARNING: pi exited with code {proc.returncode}")
    print(f"  Output saved to {out_dir.relative_to(project_root)}/")

    return {
        "test_id": "t4", "task": task, "rep": rep,
        "load_time_s": "", "ttft_ms": "",
        "total_time_s": round(total_s, 2),
        "gen_tps_burst": "", "gen_tps_sustained": "",
        "quality_score": "",
        "notes": f"output: results/t4_{runner}_{task}/ — fill quality_score manually (0-100)",
    }


def run_t5(runner: str, cfg: dict[str, str], task: str, rep: int) -> dict[str, Any]:
    prompt_file = {"email": "t5_email.txt", "news": "t5_news.txt", "pr": "t5_pr_diff.txt"}[task]
    prompt = (PROMPTS / prompt_file).read_text()
    result = chat(cfg["base_url"], cfg["model"], prompt, api_key=cfg["api_key"], max_tokens=512)
    return {
        "test_id": "t5", "task": task, "rep": rep,
        "load_time_s": "", "ttft_ms": result["ttft_ms"],
        "total_time_s": result["total_time_s"],
        "gen_tps_burst": "", "gen_tps_sustained": result["tps"],
        "quality_score": "",
    }


# ---------------------------------------------------------------------------
# Run one test with metrics capture
# ---------------------------------------------------------------------------

def run_with_metrics(runner: str, fn, *args) -> dict[str, Any]:
    run_id = f"{runner}_{int(time.time())}"
    mem = MemorySampler()
    pw_proc, pw_path = start_powermetrics(run_id)
    mem.start()

    data = fn(runner, RUNNERS[runner], *args)

    mem_stats = mem.stop()
    stop_powermetrics(pw_proc)
    if pw_path.stat().st_size == 0:
        print(f"  WARNING: {pw_path.name} is empty — powermetrics may need NOPASSWD sudo access")
    pw_stats = parse_powermetrics(pw_path)

    return {
        "run_id": run_id,
        "runner": runner,
        **data,
        **mem_stats,
        **pw_stats,
        "notes": "",
    }


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def append_csv(row: dict[str, Any]) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = raw_csv(row["runner"])
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", choices=["ollama", "omlx"], default=None,
                        help="Run against one runner only (default: both)")
    parser.add_argument("--test", choices=["t1", "t2", "t3", "t4", "t5"], default=None,
                        help="Run a single test group only")
    args = parser.parse_args()

    runners = [args.runner] if args.runner else list(RUNNERS.keys())
    tests = [args.test] if args.test else ["t1", "t2", "t3", "t4", "t5"]

    for runner in runners:
        print(f"\n=== Runner: {runner} ===")

        if "t1" in tests:
            print("T1 cold-start (restart runner between reps manually)")
            for rep in range(1, 4):
                input(f"  Rep {rep}: start the runner fresh, then press Enter...")
                row = run_with_metrics(runner, run_t1, rep)
                append_csv(row)
                print(f"  load_time={row['load_time_s']}s  ttft={row['ttft_ms']}ms")

        if "t2" in tests:
            print("T2 raw generation speed")
            for rep in range(1, 4):
                row = run_with_metrics(runner, run_t2, rep)
                append_csv(row)
                tps = row.get("gen_tps_burst") or row.get("gen_tps_sustained") or row.get("notes")
                print(f"  rep {rep}: tps={tps}  ttft={row['ttft_ms']}ms")
                if rep < 3:
                    time.sleep(120)

        if "t3" in tests:
            print("T3 TTFT / prefill")
            for rep in range(1, 4):
                row = run_with_metrics(runner, run_t3, rep)
                append_csv(row)
                print(f"  rep {rep}: ttft={row['ttft_ms']}ms")

        if "t4" in tests:
            print("T4 coding tasks (Pi agent — review output and fill quality_score manually)")
            for task in ["task_a", "task_b"]:
                row = run_with_metrics(runner, run_t4, task, 1)
                append_csv(row)
                print(f"  {task}: total={row['total_time_s']}s  quality_score=<fill manually>")

        if "t5" in tests:
            print("T5 daily-use tasks")
            for task in ["email", "news", "pr"]:
                for rep in range(1, 3):
                    row = run_with_metrics(runner, run_t5, task, rep)
                    append_csv(row)
                    print(f"  {task} rep {rep}: ttft={row['ttft_ms']}ms  total={row['total_time_s']}s")

        print(f"\nDone with {runner}. Cool down 5 minutes before next runner.")
        if runner != runners[-1]:
            time.sleep(300)

    print(f"\nAll done. Results in {RESULTS}")


if __name__ == "__main__":
    main()
