"""Supervised fine-tuning of a vision LM with Unsloth + LoRA.

Usage:
    python -m mathscribe.train.sft --config configs/train/sft_smoke.yaml
    python -m mathscribe.train.sft --config configs/train/sft_base.yaml --set tracking.enabled=true
    python -m mathscribe.train.sft --config ... --set hub.repo_id=user/repo --resume
    python -m mathscribe.train.sft --config ... --dry-run   # no GPU needed

Heavy imports (unsloth, torch, trl) happen inside ``run()`` so that the config,
resume and argument logic can be tested without a GPU stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

from mathscribe.config import SFTRunConfig, load_config
from mathscribe.cost import run_cost
from mathscribe.runinfo import git_commit, git_dirty, git_remote, package_versions

log = logging.getLogger("mathscribe.sft")

CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
# transformers' hub_strategy="checkpoint" uploads the latest checkpoint to this folder.
HUB_CHECKPOINT_DIR = "last-checkpoint"
# The Trainer does not push paths that start with "_", so the download stays local.
HUB_RESUME_DIR = "_hub_resume"

# SFTConfig values that vision fine-tuning with UnslothVisionDataCollator requires.
VISION_REQUIRED_ARGS: dict[str, Any] = {
    "remove_unused_columns": False,
    "dataset_text_field": "",
    "dataset_kwargs": {"skip_prepare_dataset": True},
}


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def build_sft_args(cfg: SFTRunConfig) -> dict[str, Any]:
    """Keyword arguments for ``trl.SFTConfig``."""
    args: dict[str, Any] = {"run_name": cfg.run_name, **cfg.train, **VISION_REQUIRED_ARGS}
    # The tracking section is the only switch for W&B (config_from_dict rejects report_to).
    args["report_to"] = ["wandb"] if cfg.tracking.enabled else "none"
    if cfg.hub.push:
        args.update(
            push_to_hub=True,
            hub_model_id=cfg.hub.repo_id,
            hub_strategy="checkpoint",
            hub_private_repo=cfg.hub.private,
        )
    return args


def best_checkpoint_info(
    best_model_checkpoint: str | None,
    best_metric: float | None,
    metric_for_best_model: str | None,
    load_best_model_at_end: bool,
) -> dict[str, Any] | None:
    """What ``run_info.json`` records about the best checkpoint (from the trainer state).

    ``loaded`` is false when the trainer could not load the best checkpoint at the end:
    the folder was on a disk that a Colab disconnect deleted. Then the final adapter is
    the last one, not the best one.
    """
    if not best_model_checkpoint:
        return None
    path = Path(best_model_checkpoint)
    metric = metric_for_best_model or "loss"
    found = path.is_dir()
    return {
        "path": str(path),
        "step": checkpoint_step(path),
        "metric": metric if metric.startswith("eval_") else f"eval_{metric}",
        "value": best_metric,
        "found": found,
        "loaded": bool(load_best_model_at_end and found),
    }


def checkpoint_step(path: Path) -> int:
    """Global step of a checkpoint: from ``trainer_state.json``, else from the folder name."""
    state = path / "trainer_state.json"
    if state.is_file():
        return int(json.loads(state.read_text())["global_step"])
    match = CHECKPOINT_RE.match(path.name)
    return int(match.group(1)) if match else -1


def find_last_local_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and CHECKPOINT_RE.match(p.name)]
    return max(candidates, key=checkpoint_step, default=None)


def download_hub_checkpoint(repo_id: str, dest: Path) -> Path | None:
    """Download ``last-checkpoint/`` from the Hub repo. Return None if there is none."""
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        files = list_repo_files(repo_id)
    except RepositoryNotFoundError:
        return None
    if not any(f.startswith(f"{HUB_CHECKPOINT_DIR}/") for f in files):
        return None
    snapshot_download(repo_id, allow_patterns=[f"{HUB_CHECKPOINT_DIR}/*"], local_dir=dest)
    return dest / HUB_CHECKPOINT_DIR


def resolve_resume_checkpoint(cfg: SFTRunConfig) -> Path:
    """Pick the most advanced checkpoint from the local output_dir and the Hub."""
    candidates: list[Path] = []
    local = find_last_local_checkpoint(cfg.output_dir)
    if local is not None:
        candidates.append(local)
    if cfg.hub.push and cfg.hub.repo_id:
        hub = download_hub_checkpoint(cfg.hub.repo_id, cfg.output_dir / HUB_RESUME_DIR)
        if hub is not None:
            candidates.append(hub)
    if not candidates:
        raise SystemExit(
            f"--resume: no checkpoint in {cfg.output_dir} "
            f"or in the Hub repo {cfg.hub.repo_id or '(hub.push is off)'}"
        )
    best = max(candidates, key=checkpoint_step)
    log.info("Resuming from %s (step %d)", best, checkpoint_step(best))
    return best


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mathscribe.train.sft", description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, type=Path, help="YAML run config")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set train.max_steps=30 (repeatable)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint (local output_dir or Hub last-checkpoint)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the resolved config and SFT args, then exit"
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def load_validation(cfg: SFTRunConfig) -> Any:
    """The validation rows (``validation.manifest``, first ``samples`` rows), or None."""
    if cfg.validation.manifest is None:
        return None
    from mathscribe.data import as_conversations
    from mathscribe.eval.testsets import Manifest, load_testset

    manifest = Manifest.load(cfg.validation.manifest)
    if (manifest.dataset, manifest.split) != (cfg.data.dataset, cfg.data.split):
        raise SystemExit(
            f"validation.manifest is for {manifest.dataset}/{manifest.split}, "
            f"not the training data {cfg.data.dataset}/{cfg.data.split}"
        )
    rows = load_testset(manifest, cfg.validation.samples)
    return as_conversations(rows, cfg.data.instruction)


def run(cfg: SFTRunConfig, resume_from: Path | None = None) -> dict[str, Any]:
    # Unsloth must be imported before transformers / trl / peft.
    import unsloth  # noqa: F401

    # isort: split
    import torch
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator

    from mathscribe.data import load_latex_ocr
    from mathscribe.train.run_state import make_run_state_callback, read_run_state

    model, tokenizer = FastVisionModel.from_pretrained(
        cfg.model.name,
        load_in_4bit=cfg.model.load_in_4bit,
        use_gradient_checkpointing=cfg.model.use_gradient_checkpointing,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg.lora.finetune_vision_layers,
        finetune_language_layers=cfg.lora.finetune_language_layers,
        finetune_attention_modules=cfg.lora.finetune_attention_modules,
        finetune_mlp_modules=cfg.lora.finetune_mlp_modules,
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        use_rslora=cfg.lora.use_rslora,
        loftq_config=None,
        random_state=cfg.lora.random_state,
    )

    train_dataset = load_latex_ocr(cfg.data)
    log.info("Train dataset: %s/%s, %d rows", cfg.data.dataset, cfg.data.split, len(train_dataset))
    eval_dataset = load_validation(cfg)
    if eval_dataset is not None:
        log.info("Validation: %s, %d rows", cfg.validation.manifest, len(eval_dataset))

    callbacks: list[Any] = []
    if cfg.validation.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.validation.early_stopping_patience,
                early_stopping_threshold=cfg.validation.early_stopping_threshold,
            )
        )
    # A resumed run continues the W&B run and the train time stored in the checkpoint.
    prior = read_run_state(resume_from)
    wandb_run = None
    if cfg.tracking.enabled:
        from mathscribe import tracking

        wandb_run = tracking.init_run(
            cfg.tracking,
            name=cfg.run_name,
            job_type="train",
            config=cfg.to_dict(),
            group=cfg.run_name,
            run_id=prior["wandb_run_id"],
        )
    callbacks.append(
        make_run_state_callback(wandb_run.id if wandb_run else None, prior["train_seconds"])
    )

    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        args=SFTConfig(**build_sft_args(cfg)),
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    gpu_props = torch.cuda.get_device_properties(0)
    start_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    wall_clock_s = time.time() - t0

    best = best_checkpoint_info(
        trainer.state.best_model_checkpoint,
        trainer.state.best_metric,
        trainer.args.metric_for_best_model,
        trainer.args.load_best_model_at_end,
    )
    if best is not None and trainer.args.load_best_model_at_end and not best["loaded"]:
        log.warning(
            "The best checkpoint %s does not exist on this disk, so the final adapter is the "
            "LAST one, not the best one. Use USE_DRIVE = True on Colab for runs with "
            "load_best_model_at_end.",
            best["path"],
        )

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    info: dict[str, Any] = {
        "run_name": cfg.run_name,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "git_remote": git_remote(),
        "resumed_from": str(resume_from) if resume_from else None,
        "train_rows": len(train_dataset),
        "eval_rows": len(eval_dataset) if eval_dataset is not None else 0,
        "best_checkpoint": best,
        "wandb": None,
        "gpu": gpu_props.name,
        "gpu_total_memory_gb": round(gpu_props.total_memory / 1024**3, 3),
        "peak_reserved_memory_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "reserved_memory_before_train_gb": round(start_reserved_gb, 3),
        "wall_clock_s": round(wall_clock_s, 1),  # this session only
        # All sessions: the earlier ones up to the resume checkpoint, plus this one.
        "total_train_s": round(prior["train_seconds"] + wall_clock_s, 1),
        "cost": run_cost(gpu_props.name, prior["train_seconds"] + wall_clock_s),
        "global_step": trainer.state.global_step,
        "train_metrics": stats.metrics,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "python": platform.python_version(),
        "versions": package_versions(
            ["mathscribe", "unsloth", "transformers", "trl", "peft", "torch", "datasets"]
        ),
        "config": cfg.to_dict(),
    }
    if wandb_run is not None:
        from mathscribe import tracking

        info["wandb"] = tracking.run_ref(wandb_run)
    run_info_path = cfg.output_dir / "run_info.json"
    run_info_path.write_text(json.dumps(info, indent=2, default=str))
    log.info("Run info: %s", run_info_path)

    from mathscribe.train.model_card import push_model_card, write_model_card

    card = cfg.output_dir / "README.md"
    if cfg.hub.push:
        trainer.push_to_hub(commit_message=f"{cfg.run_name}: final (step {info['global_step']})")
        log.info("Pushed final adapter to https://huggingface.co/%s", cfg.hub.repo_id)
        # push_to_hub writes the Trainer's own README.md. Replace it with the MathScribe card.
        write_model_card(info, card, repo_id=cfg.hub.repo_id)
        push_model_card(card, cfg.hub.repo_id, f"{cfg.run_name}: model card")
    else:
        write_model_card(info, card)

    if wandb_run is not None:
        from mathscribe import tracking

        wandb_run.summary.update(
            tracking.flatten(
                "run_info",
                {k: v for k, v in info.items() if k not in ("config", "versions", "train_metrics")},
            )
        )
        wandb_run.save(str(run_info_path), base_path=str(cfg.output_dir), policy="now")
        wandb_run.finish()
    return info


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)

    if args.dry_run:
        print(json.dumps({"config": cfg.to_dict(), "sft_args": build_sft_args(cfg)}, indent=2))
        return

    resume_from = resolve_resume_checkpoint(cfg) if args.resume else None
    run(cfg, resume_from)


if __name__ == "__main__":
    sys.exit(main())
