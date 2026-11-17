"""W&B logging for training and evaluation.

Training: ``init_run`` starts the W&B run before the trainer starts. The transformers
``WandbCallback`` then logs into this run (it uses ``wandb.run`` when it exists).
The run ID goes into each checkpoint (``mathscribe.train.run_state``), so a ``--resume``
on a new Colab session continues the same W&B run.

Evaluation: ``log_eval_result`` writes the metrics of one eval run to the W&B summary.

``wandb`` is imported only in the functions that need it, so the tests need no ``wandb``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mathscribe.config import TrackingConfig

log = logging.getLogger("mathscribe.tracking")


def ensure_login() -> None:
    """Stop with a clear message when no W&B API key is set (instead of an input prompt)."""
    import wandb

    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return  # offline runs need no API key (sync them later with `wandb sync`)
    try:
        ok = wandb.login(prompt=False, verify=False)
    except wandb.errors.UsageError as e:
        ok, reason = False, str(e)
    else:
        reason = "no API key"
    if not ok:
        raise SystemExit(
            f"tracking.enabled is true, but W&B has no login ({reason}). "
            "Set WANDB_API_KEY (Colab: the WANDB_API_KEY secret), or run `wandb login`, "
            "or set tracking.enabled=false."
        )


def init_run(
    cfg: TrackingConfig,
    *,
    name: str,
    job_type: str,
    config: dict[str, Any],
    group: str | None = None,
    run_id: str | None = None,
) -> Any:
    """Start a W&B run (or continue the run ``run_id``). Return the run."""
    import wandb

    ensure_login()
    run = wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        name=name,
        group=cfg.group or group,
        job_type=job_type,
        tags=list(cfg.tags),
        config=config,
        id=run_id,
        resume="allow" if run_id else None,
    )
    log.info("W&B run %s: %s", "continued" if run_id else "started", run.url)
    return run


def run_ref(run: Any) -> dict[str, Any] | None:
    """The fields that results, run_info.json and model cards store about a W&B run."""
    if run is None:
        return None
    return {"id": run.id, "url": run.url, "project": run.project, "entity": run.entity}


def flatten(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    """``{"a": {"b": 1}}`` -> ``{"prefix/a/b": 1}``. Lists stay as they are."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(name, value))
        else:
            out[name] = value
    return out


def eval_summary(result: dict[str, Any]) -> dict[str, Any]:
    """The W&B summary of one eval result (``run_eval`` result dict, without per-row data)."""
    summary = flatten("eval", result["metrics"])
    for metric, (low, high) in result.get("ci95", {}).items():
        summary[f"eval_ci95/{metric}_low"] = low
        summary[f"eval_ci95/{metric}_high"] = high
    for tag, metrics in result.get("buckets", {}).items():
        summary.update(flatten(f"eval_bucket/{tag}", metrics))
    runtime = {k: v for k, v in result.get("runtime", {}).items() if not isinstance(v, dict)}
    summary.update(flatten("runtime", runtime))
    summary.update(flatten("cost", result.get("runtime", {}).get("cost") or {}))
    summary["testset/name"] = result["testset"]["name"]
    summary["testset/n"] = result["testset"]["n"]
    summary["testset/sha256"] = result["testset"]["sha256"]
    return summary


def log_eval_result(run: Any, result: dict[str, Any], predictions: Path | None = None) -> None:
    """Write an eval result to the run summary. Upload ``predictions.jsonl`` as a file."""
    run.summary.update(eval_summary(result))
    if predictions is not None and predictions.is_file():
        run.save(str(predictions), base_path=str(predictions.parent), policy="now")
