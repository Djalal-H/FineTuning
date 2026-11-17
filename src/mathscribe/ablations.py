"""Run the Phase 2 ablations on one or more GPUs.

Each run in ``configs/ablations.yaml`` is one job: a training step, then (optionally) an
evaluation of the adapter on the val set. The jobs share a queue. Each GPU has one worker,
and each worker runs one job at a time in its own process (``CUDA_VISIBLE_DEVICES`` is set
for each step). Unsloth uses one GPU for each process, so two GPUs give two jobs in parallel.

Usage:
    python -m mathscribe.ablations --dry-run                  # check every config, no GPU
    python -m mathscribe.ablations --gpus 0,1                 # all runs on two GPUs
    python -m mathscribe.ablations --gpus 0 --runs base,dora  # two runs on one GPU

A step is skipped when its result file exists (``run_info.json`` for training, the result
JSON for the eval). So run the same command again after a stop: finished steps are skipped,
and training continues from the last checkpoint (local, or the Hub ``last-checkpoint/``).
A failed job does not stop the other jobs.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

from mathscribe.config import ConfigError, load_config

SPEC = "configs/ablations.yaml"
EVAL_CONFIG = "configs/eval/sft_base.yaml"
VAL_MANIFEST = "data/testsets/latex_ocr_val.json"


@dataclasses.dataclass
class Step:
    label: str
    command: Callable[[], list[str]]  # called when the step starts (the files may be new then)
    done_file: Path


@dataclasses.dataclass
class Job:
    name: str
    steps: list[Step]


@dataclasses.dataclass
class JobResult:
    name: str
    status: str  # "done", "skipped" or "FAILED"
    seconds: float
    gpu: int | None = None
    failed_step: str | None = None


# --------------------------------------------------------------------------- #
# Build the jobs
# --------------------------------------------------------------------------- #
def load_spec(path: str | Path = SPEC) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text())
    for key in ("base_config", "hub_prefix", "runs"):
        if key not in spec:
            raise ConfigError(f"{path}: missing key {key!r}")
    names = [run["name"] for run in spec["runs"]]
    if len(names) != len(set(names)):
        raise ConfigError(f"{path}: run names must be unique, got {names}")
    return spec


def select_runs(spec: dict[str, Any], names: Sequence[str] | None) -> list[dict[str, Any]]:
    """The runs to do, in the order of the spec. ``names=None`` selects all runs."""
    runs = spec["runs"]
    if names is None:
        return list(runs)
    known = {run["name"] for run in runs}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ConfigError(f"Unknown ablation runs {unknown}. Known runs: {sorted(known)}")
    return [run for run in runs if run["name"] in set(names)]


def _set_args(overrides: Sequence[str]) -> list[str]:
    return [arg for o in overrides for arg in ("--set", o)]


def build_jobs(
    spec: dict[str, Any],
    names: Sequence[str] | None,
    *,
    python: str,
    output_root: str | Path,
    results_dir: str | Path,
    tracking: bool = True,
    extra_overrides: Sequence[str] = (),
    tag: str = "",
    hub: bool = True,
    evaluate: bool = True,
    eval_rows: int = 500,
) -> list[Job]:
    """One job for each selected run. Every training config is loaded here (no GPU needed),
    so a wrong override stops the program before any GPU time is used."""
    output_root, results_dir = Path(output_root), Path(results_dir)
    suffix = f"_{tag}" if tag else ""
    base_config = spec["base_config"]
    if evaluate:
        manifest = json.loads(Path(VAL_MANIFEST).read_text())
        testset = manifest["name"]
        n_rows = min(eval_rows, len(manifest["rows"]))
    baseline = f"abl_base{suffix}"

    jobs = []
    for run in select_runs(spec, names):
        name = f"abl_{run['name']}{suffix}"
        out = output_root / name
        repo = f"{spec['hub_prefix']}-{run['name']}{suffix}"
        overrides = [
            f"run_name={name}",
            f"train.output_dir={out}",
            f"tracking.enabled={'true' if tracking else 'false'}",
            *extra_overrides,
            *run["overrides"],
        ]
        if hub:
            overrides += ["hub.push=true", f"hub.repo_id={repo}"]
        cfg = load_config(base_config, overrides)  # raises ConfigError on a bad override

        train_cmd = [python, "-m", "mathscribe.train.sft", "--config", base_config]
        train_cmd += ["--resume-if-available", *_set_args(overrides)]
        steps = [Step("train", lambda c=train_cmd: c, out / "run_info.json")]

        if evaluate:

            def eval_cmd(
                name=name,
                out=out,
                repo=repo,
                run=run,
                four_bit=cfg.model.load_in_4bit,
            ) -> list[str]:
                # The adapter on this disk, or (a new session) the final adapter on the Hub.
                model = str(out) if (out / "adapter_config.json").exists() else repo
                eval_overrides = [
                    f"run_name={name}",
                    f"model.name={model}",
                    f"model.label=Ablation {run['factor']}",
                    f"model.load_in_4bit={'true' if four_bit else 'false'}",
                    f"testset.manifest={VAL_MANIFEST}",
                    f"output.dir={output_root / 'eval'}",
                    f"output.results_dir={results_dir}",
                    f"output.leaderboard={output_root / 'eval' / name / 'leaderboard.md'}",
                    f"output.baseline_run={baseline}",
                    f"render.cache_dir={output_root / 'render_cache' / name}",
                    f"tracking.enabled={'true' if tracking else 'false'}",
                ]
                return [
                    python,
                    "-m",
                    "mathscribe.eval.run_eval",
                    "--config",
                    EVAL_CONFIG,
                    "--subset",
                    str(n_rows),
                    *_set_args(eval_overrides),
                ]

            steps.append(Step("eval", eval_cmd, results_dir / f"{name}__{testset}_n{n_rows}.json"))
        jobs.append(Job(name, steps))
    return jobs


# --------------------------------------------------------------------------- #
# Run the jobs
# --------------------------------------------------------------------------- #
_PRINT_LOCK = threading.Lock()
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


def terminate_all() -> None:
    """Stop every running step (and its child processes)."""
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    for proc in procs:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)


def run_step(command: list[str], gpu: int, prefix: str, log_file: Path, cache_root: Path) -> int:
    """Run one command on one GPU. Echo its output with a prefix, and keep a log file."""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        # One Unsloth compile cache for each GPU: two processes must not write the same files.
        "UNSLOTH_COMPILE_LOCATION": str((cache_root / f"gpu{gpu}").resolve()),
        "TQDM_MININTERVAL": os.environ.get("TQDM_MININTERVAL", "5"),
    }
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as log:
        log.write(f"$ {' '.join(command)}\n")
        proc = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,  # a process group, so that terminate_all() reaches the children
        )
        with _ACTIVE_LOCK:
            _ACTIVE.add(proc)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                with _PRINT_LOCK:
                    print(f"[{prefix}] {line}", end="", flush=True)
            return proc.wait()
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(proc)


def run_jobs(
    jobs: Sequence[Job], gpus: Sequence[int], *, log_dir: str | Path, cache_root: str | Path
) -> list[JobResult]:
    """Run the jobs on the GPUs: one worker for each GPU, one job at a time for each worker."""
    log_dir, cache_root = Path(log_dir), Path(cache_root)
    todo: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        todo.put(job)
    results: dict[str, JobResult] = {}

    def worker(gpu: int) -> None:
        while True:
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            status, failed = "skipped", None
            for step in job.steps:
                if step.done_file.exists():
                    continue
                with _PRINT_LOCK:
                    print(f"===== gpu{gpu}: {job.name} {step.label} =====", flush=True)
                code = run_step(
                    step.command(), gpu, f"gpu{gpu} {job.name}", log_dir / f"{job.name}.log",
                    cache_root,
                )  # fmt: skip
                if code != 0:
                    status, failed = "FAILED", step.label
                    break
                status = "done"
            results[job.name] = JobResult(job.name, status, time.time() - t0, gpu, failed)

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    try:
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)
    except KeyboardInterrupt:
        terminate_all()
        raise
    return [results[j.name] for j in jobs if j.name in results]


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mathscribe.ablations", description=__doc__.splitlines()[0])
    p.add_argument("--spec", default=SPEC, help="the ablation list (default: %(default)s)")
    p.add_argument("--runs", help="comma-separated run names (default: all runs in the spec)")
    p.add_argument("--gpus", default="0", help="GPU ids, comma-separated (default: %(default)s)")
    p.add_argument("--output-root", default="outputs", help="adapters, logs and eval files")
    p.add_argument("--results-dir", default="reports/results", help="the eval result JSONs")
    p.add_argument("--eval-rows", type=int, default=500, help="the first N val rows (default 500)")
    p.add_argument("--no-eval", action="store_true", help="train only")
    p.add_argument("--no-hub", action="store_true", help="do not push to the Hub (no resume)")
    p.add_argument("--no-tracking", action="store_true", help="no W&B")
    p.add_argument("--tag", default="", help="suffix for the run names, for example 'smoke'")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Extra training override for every run, e.g. --set train.max_steps=10",
    )  # fmt: skip
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = load_spec(args.spec)
    names = [n.strip() for n in args.runs.split(",") if n.strip()] if args.runs else None
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus needs at least one GPU id")
    root = Path(args.output_root)
    jobs = build_jobs(
        spec,
        names,
        python=sys.executable,
        output_root=root,
        results_dir=args.results_dir,
        tracking=not args.no_tracking,
        extra_overrides=args.overrides,
        tag=args.tag,
        hub=not args.no_hub,
        evaluate=not args.no_eval,
        eval_rows=args.eval_rows,
    )

    if args.dry_run:
        for job in jobs:
            print(f"# {job.name}")
            for step in job.steps:
                print(f"{step.label} (done file: {step.done_file}):\n  {' '.join(step.command())}")
        return 0

    workers = min(len(gpus), len(jobs)) or 1
    print(f"{len(jobs)} job(s) on GPU(s) {gpus[:workers]}", flush=True)
    results = run_jobs(jobs, gpus[:workers], log_dir=root / "logs", cache_root=root / "cache")
    print("\n===== Summary =====")
    for r in results:
        extra = f" (failed at {r.failed_step})" if r.failed_step else ""
        print(f"{r.status:8} {r.seconds / 60:6.1f} min  gpu{r.gpu}  {r.name}{extra}")
    return 1 if any(r.status == "FAILED" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
