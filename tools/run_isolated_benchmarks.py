"""Isolated GPU benchmarks with power, memory, and repeatability.

Refuses to run while the trainer or live runtime is active, samples
nvidia-smi every 100 ms during each run, repeats every benchmark, and writes
a JSON record plus a Markdown summary (median, range, coefficient of
variation). Throughput comes from the benchmarks' own output; power, memory,
utilization, temperature, and clocks come from the sampler.

    .venv\\Scripts\\python tools\\run_isolated_benchmarks.py --repeats 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BLOCKING_PROCESSES = ("t8_v2_train.exe", "t8_v2_visualizer_feed.exe")
BLOCKING_COMMAND_LINES = ("live_vision_play.py", "visualize_v2.py")
SAMPLE_FIELDS = ("power.draw", "memory.used", "utilization.gpu", "temperature.gpu", "clocks.sm")


def benchmarks(build: Path) -> list[dict]:
    simulator = str(build / "t8_v2_gpu_benchmark.exe")
    training = str(build / "t8_v2_training_benchmark.exe")
    return [
        {"name": "simulator", "command": [simulator, "--envs", "262144", "--steps", "2000"]},
        # The README configuration: short, so throughput reflects cold-ish
        # updates and power has few samples.
        {"name": "training_readme", "command": [
            training, "--envs", "4096", "--horizon", "128", "--updates", "5",
            "--minibatch", "4096", "--epochs", "4", "--visual"]},
        # The overnight run's shape, long enough to reach steady power/clocks.
        {"name": "training_sustained", "command": [
            training, "--envs", "32768", "--horizon", "128", "--updates", "20",
            "--minibatch", "131072", "--epochs", "4", "--visual"]},
    ]


def run_text(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def blocking_processes() -> list[str]:
    listing = run_text(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process | ForEach-Object { $_.Name + '|' + $_.CommandLine }"])
    found = []
    for line in listing.splitlines():
        name, _, command_line = line.partition("|")
        if name.lower() in BLOCKING_PROCESSES or any(marker in command_line for marker in BLOCKING_COMMAND_LINES):
            found.append(line.strip())
    return found


def gpu_compute_apps() -> list[str]:
    output = run_text(["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader"])
    return [line.strip() for line in output.splitlines() if line.strip()]


class Sampler:
    """Polls nvidia-smi on a background thread."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.samples: list[dict[str, float]] = []
        self._process = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={','.join(SAMPLE_FIELDS)}",
             "--format=csv,noheader,nounits", f"-lms={interval_ms}"],
            stdout=subprocess.PIPE, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            values = [value.strip() for value in line.split(",")]
            try:
                self.samples.append({field: float(value) for field, value in zip(SAMPLE_FIELDS, values)})
            except ValueError:
                continue

    def stop(self) -> list[dict[str, float]]:
        self._process.terminate()
        self._process.wait(timeout=10)
        self._thread.join(timeout=5)
        return self.samples


def parse_metrics(output: str) -> dict[str, float]:
    metrics = {}
    for line in output.splitlines():
        match = re.match(r"\s*([A-Za-z][A-Za-z /()-]*?):\s*([-0-9.eE+]+)\s*$", line)
        if match:
            key = re.sub(r"[^a-z0-9]+", "_", match.group(1).lower()).strip("_")
            metrics[key] = float(match.group(2))
    return metrics


def summarize(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "cv_percent": (statistics.stdev(values) / mean * 100.0) if len(values) > 1 and mean else 0.0,
    }


def measure(benchmark: dict, repeats: int, idle_memory: float) -> dict:
    runs = []
    for repeat in range(repeats):
        sampler = Sampler()
        time.sleep(0.3)  # let the sampler start before the workload
        started = time.perf_counter()
        output = run_text(benchmark["command"])
        wall = time.perf_counter() - started
        time.sleep(0.3)
        samples = sampler.stop()
        busy = [sample for sample in samples if sample["utilization.gpu"] >= 50.0] or samples
        runs.append({
            "repeat": repeat + 1,
            "wall_seconds": wall,
            "metrics": parse_metrics(output),
            "samples": len(samples),
            "busy_samples": len(busy),
            "mean_busy_power_w": statistics.fmean(sample["power.draw"] for sample in busy),
            "peak_power_w": max(sample["power.draw"] for sample in samples),
            "peak_memory_mib": max(sample["memory.used"] for sample in samples),
            "peak_memory_over_idle_mib": max(sample["memory.used"] for sample in samples) - idle_memory,
            "mean_busy_utilization_percent": statistics.fmean(sample["utilization.gpu"] for sample in busy),
            "peak_temperature_c": max(sample["temperature.gpu"] for sample in samples),
            "mean_busy_sm_clock_mhz": statistics.fmean(sample["clocks.sm"] for sample in busy),
        })
        print(f"{benchmark['name']} {repeat + 1}/{repeats}: "
              + " ".join(f"{key}={value:.4g}" for key, value in runs[-1]["metrics"].items()
                         if key.endswith("_s") or "decisions" in key or "visits" in key)
              + f" power={runs[-1]['mean_busy_power_w']:.0f}W", flush=True)
    metric_names = sorted(set.intersection(*(set(run["metrics"]) for run in runs)))
    return {
        "name": benchmark["name"],
        "command": " ".join(Path(benchmark["command"][0]).name if index == 0 else part
                            for index, part in enumerate(benchmark["command"])),
        "runs": runs,
        "summary": {
            **{name: summarize([run["metrics"][name] for run in runs]) for name in metric_names},
            **{name: summarize([run[name] for run in runs]) for name in (
                "wall_seconds", "mean_busy_power_w", "peak_power_w", "peak_memory_mib",
                "peak_memory_over_idle_mib", "mean_busy_utilization_percent",
                "peak_temperature_c", "mean_busy_sm_clock_mhz")},
        },
    }


def environment() -> dict:
    gpu = run_text(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                    "--format=csv,noheader"]).strip()
    cuda = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", run_text(["nvidia-smi"]))
    cpu = run_text(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]).strip()
    commit = run_text(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).strip()
    dirty = bool(run_text(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"]).strip())
    return {
        "gpu": gpu, "cuda_driver_api": cuda.group(1) if cuda else "unknown", "cpu": cpu,
        "os": platform.platform(), "commit": commit + ("+uncommitted" if dirty else ""),
    }


def markdown(record: dict) -> str:
    lines = [
        f"# Isolated benchmarks ({record['date']})", "",
        f"- GPU: {record['environment']['gpu']} (CUDA driver API {record['environment']['cuda_driver_api']})",
        f"- CPU: {record['environment']['cpu']}",
        f"- OS: {record['environment']['os']}",
        f"- Build: Release, commit {record['environment']['commit']}",
        f"- Isolation: no trainer, visualizer, or live runtime running; "
        f"{record['other_gpu_clients']} desktop applications held GPU contexts",
        f"- Idle before runs: {record['idle']['power_w']:.1f} W, {record['idle']['memory_mib']:.0f} MiB used",
        f"- {record['repeats']} repeats per benchmark; power/memory sampled every 100 ms with nvidia-smi "
        "(power averaged over samples at >=50% utilization). Runs of about one second yield only "
        "around 10 busy samples, so their power figures are indicative; `training_sustained` "
        "(about 30 s) is the steady-state power and memory measurement.", "",
        "| Benchmark | Metric | Median | Range | CV |", "|---|---|---:|---:|---:|",
    ]
    wanted = ("environment_decisions_s", "simulated_frames_s", "ppo_sample_visits_s", "total_seconds",
              "mean_busy_power_w", "peak_power_w", "peak_memory_over_idle_mib", "peak_temperature_c",
              "mean_busy_sm_clock_mhz")
    for benchmark in record["benchmarks"]:
        for name in wanted:
            stats = benchmark["summary"].get(name)
            if not stats:
                continue
            lines.append(f"| {benchmark['name']} | {name} | {stats['median']:,.4g} | "
                         f"{stats['min']:,.4g} – {stats['max']:,.4g} | {stats['cv_percent']:.1f}% |")
    lines += ["", "Commands:", ""] + [f"- `{benchmark['name']}`: `{benchmark['command']}`"
                                      for benchmark in record["benchmarks"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", type=Path, default=REPO / "build-gpu" / "Release")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs" / "benchmarks")
    args = parser.parse_args()
    blockers = blocking_processes()
    if blockers:
        raise SystemExit("refusing to benchmark while these are running:\n  " + "\n  ".join(blockers))
    apps_before = gpu_compute_apps()
    sampler = Sampler()
    time.sleep(5)
    idle = sampler.stop()
    record = {
        "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "environment": environment(),
        "other_gpu_clients": len(apps_before),
        "idle": {"power_w": statistics.median(sample["power.draw"] for sample in idle),
                 "memory_mib": statistics.median(sample["memory.used"] for sample in idle)},
        "repeats": args.repeats,
        "benchmarks": [],
    }
    for benchmark in benchmarks(args.build):
        record["benchmarks"].append(measure(benchmark, args.repeats, record["idle"]["memory_mib"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "isolated_" + dt.datetime.now().strftime("%Y-%m-%d")
    (args.output_dir / f"{stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (args.output_dir / f"{stem}.md").write_text(markdown(record), encoding="utf-8")
    print(f"wrote {args.output_dir / (stem + '.json')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
