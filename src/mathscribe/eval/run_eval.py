"""Evaluate a model (or a baseline) on a frozen test set and update the leaderboard.

Usage:
    python -m mathscribe.eval.run_eval --config configs/eval/qwen35_4b_zeroshot.yaml
    python -m mathscribe.eval.run_eval --config configs/eval/sft_smoke.yaml \\
        --set model.name=<hf-user>/mathscribe-sft-smoke --subset 100
    # Score the stored outputs of a run again (no GPU):
    python -m mathscribe.eval.run_eval --config configs/eval/qwen35_4b_zeroshot.yaml \\
        --set model.backend=jsonl --set model.name=outputs/eval/.../predictions.jsonl

Each run writes:
    <output.dir>/<run_name>/<testset>_n<N>/predictions.jsonl   one line per row
    <output.dir>/<run_name>/<testset>_n<N>/metrics.json        the result, as below
    <output.results_dir>/<run_name>__<testset>_n<N>.json        the result (commit this)
    <output.leaderboard>                                       built from results_dir
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mathscribe.config import ConfigError
from mathscribe.eval.bootstrap import per_row_scores, run_ci
from mathscribe.eval.config import EvalRunConfig, load_eval_config
from mathscribe.eval.leaderboard import write_leaderboard
from mathscribe.eval.metrics import aggregate, aggregate_by_tag, difficulty_tags, score_sample
from mathscribe.eval.normalize import extract_latex
from mathscribe.eval.testsets import Manifest, load_testset, subset_hash
from mathscribe.runinfo import git_commit, package_versions

if TYPE_CHECKING:
    from datasets import Dataset

    from mathscribe.eval.predictors import Predictor

log = logging.getLogger("mathscribe.eval")

LOG_EVERY_S = 30.0


def evaluate(
    predictor: Predictor, rows: Dataset, batch_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the predictor on all rows. Return the per-row records and the summary."""
    records: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    tags: list[list[str]] = []
    t0 = last_log = time.time()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        batch_rows = [
            {"image": image, "source_index": index}
            for image, index in zip(batch["image"], batch["source_index"], strict=True)
        ]
        raws = predictor.predict(batch_rows)
        if len(raws) != len(batch_rows):
            raise RuntimeError(f"predictor returned {len(raws)} outputs for {len(batch_rows)} rows")
        for index, reference, raw in zip(batch["source_index"], batch["text"], raws, strict=True):
            prediction = extract_latex(raw)
            score = score_sample(prediction, reference)
            sample_tags = difficulty_tags(reference)
            scores.append(score)
            tags.append(sample_tags)
            records.append(
                {
                    "source_index": index,
                    "reference": reference,
                    "raw": raw,
                    "prediction": prediction,
                    "exact_match": score["exact_match"],
                    "normalized_match": score["normalized_match"],
                    "cer": round(score["cer"], 4),
                    "tags": sample_tags,
                }
            )
        now = time.time()
        if now - last_log >= LOG_EVERY_S or len(records) == len(rows):
            done = len(records)
            eta = (now - t0) / done * (len(rows) - done)
            log.info(
                "%d/%d rows, normalized match so far %.1f%%, ETA %.0f s",
                done,
                len(rows),
                100 * aggregate(scores)["normalized_match"],
                eta,
            )
            last_log = now
    wall_clock_s = time.time() - t0
    per_row = per_row_scores(scores)
    summary = {
        "metrics": aggregate(scores),
        "ci95": run_ci(per_row) if scores else {},
        "buckets": aggregate_by_tag(scores, tags),
        "per_row": per_row,
        "wall_clock_s": round(wall_clock_s, 1),
        "images_per_s": round(len(rows) / wall_clock_s, 3) if wall_clock_s > 0 else None,
    }
    return records, summary


def run_paths(cfg: EvalRunConfig, testset_name: str, n: int) -> tuple[Path, Path]:
    tag = f"{testset_name}_n{n}"
    run_dir = Path(cfg.output.dir) / cfg.run_name / tag
    result_file = Path(cfg.output.results_dir) / f"{cfg.run_name}__{tag}.json"
    return run_dir, result_file


def run(
    cfg: EvalRunConfig, predictor: Predictor | None = None, source: Dataset | None = None
) -> dict[str, Any]:
    """Evaluate and write all outputs. ``predictor`` and ``source`` are for tests."""
    manifest = Manifest.load(cfg.testset.manifest)
    rows = load_testset(manifest, cfg.testset.subset, source=source)
    indices, hashes = manifest.prefix(cfg.testset.subset)
    log.info("Test set %s: %d of %d rows", manifest.name, len(rows), len(manifest))

    if predictor is None:
        from mathscribe.eval.predictors import build_predictor

        predictor = build_predictor(cfg.model, cfg.generation)

    records, summary = evaluate(predictor, rows, cfg.generation.batch_size)

    # A replayed run (backend jsonl) keeps the timing of the original run.
    runtime = {"wall_clock_s": summary["wall_clock_s"], "images_per_s": summary["images_per_s"]}
    runtime.update(predictor.runtime_info())
    result = {
        "run_name": cfg.run_name,
        "label": cfg.label,
        "date": datetime.date.today().isoformat(),
        "git_commit": git_commit(),
        "testset": {
            "name": manifest.name,
            "manifest": cfg.testset.manifest,
            "dataset": manifest.dataset,
            "revision": manifest.revision,
            "split": manifest.split,
            "n": len(indices),
            "sha256": subset_hash(indices, hashes),
        },
        "metrics": summary["metrics"],
        "ci95": summary["ci95"],
        "buckets": summary["buckets"],
        "runtime": runtime,
        "python": platform.python_version(),
        "versions": package_versions(
            ["mathscribe", "unsloth", "transformers", "torch", "datasets"]
        ),
        "config": cfg.to_dict(),
        # Per-row scores in the test set order, for the paired intervals of the leaderboard.
        "per_row": summary["per_row"],
    }

    run_dir, result_file = run_paths(cfg, manifest.name, len(indices))
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    text = json.dumps(result, indent=2, default=str) + "\n"
    (run_dir / "metrics.json").write_text(text)
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(text)
    board = write_leaderboard(
        cfg.output.results_dir, cfg.output.leaderboard, cfg.output.baseline_run
    )

    m = result["metrics"]
    lo, hi = result["ci95"]["normalized_match"]
    log.info(
        "%s: exact %.1f%%, normalized %.1f%% (95%% CI %.1f-%.1f), CER %.1f%%, BLEU %.1f (n=%d)",
        cfg.label,
        100 * m["exact_match"],
        100 * m["normalized_match"],
        100 * lo,
        100 * hi,
        100 * m["cer"],
        100 * m["bleu"],
        m["n"],
    )
    log.info("Predictions: %s", run_dir / "predictions.jsonl")
    log.info("Result: %s. Leaderboard: %s", result_file, board)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="mathscribe.eval.run_eval", description=__doc__.splitlines()[0]
    )
    p.add_argument("--config", required=True, type=Path, help="YAML eval config")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set generation.batch_size=4 (repeatable)",
    )
    p.add_argument("--subset", type=int, default=None, help="Use the first N frozen rows")
    p.add_argument("--dry-run", action="store_true", help="Print the resolved config, then exit")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    overrides = list(args.overrides)
    if args.subset is not None:
        overrides.append(f"testset.subset={args.subset}")
    try:
        cfg = load_eval_config(args.config, overrides)
    except ConfigError as e:
        raise SystemExit(f"Config error: {e}") from e
    if args.dry_run:
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    run(cfg)


if __name__ == "__main__":
    main()
