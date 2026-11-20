import json

import yaml

from mathscribe.ablation_report import load_ablations, render_report, write_report

SPEC = {
    "base_config": "configs/train/ablation_base.yaml",
    "hub_prefix": "user/abl",
    "runs": [
        {"name": "base", "factor": "LoRA r=16 (base)", "overrides": []},
        {"name": "r8", "factor": "r=8", "overrides": ["lora.r=8"]},
        {"name": "worse", "factor": "worse", "overrides": ["lora.r=2"]},
    ],
    "skipped": [
        {
            "name": "dora",
            "factor": "DoRA",
            "overrides": ["lora.use_dora=true"],
            "reason": "nan loss\n  from step 1",
        }
    ],
}


def _flags(n_ones: int, n: int = 100) -> str:
    return "1" * n_ones + "0" * (n - n_ones)


def _result(name: str, nm_ones: int, n: int = 100) -> dict:
    flags = _flags(nm_ones, n)
    return {
        "run_name": name,
        "label": name,
        "scorer_version": 2,
        "renderer": "r1",
        "testset": {"name": "latex_ocr_val", "n": n, "sha256": "a" * 64},
        "metrics": {
            "normalized_match": nm_ones / n,
            "render_match": nm_ones / n,
            "compile_rate": 1.0,
            "cer": 0.03,
        },
        "runtime": {"images_per_s": 0.6},
        "wandb": {"url": f"https://wandb.example/{name}-eval"},
        "per_row": {
            "exact_match": flags,
            "normalized_match": flags,
            "char_edits": [0] * n,
            "ref_chars": [10] * n,
            "compiles": "1" * n,
            "render_match": flags,
        },
    }


def _info(name: str, resumed: bool = False) -> dict:
    return {
        "run_name": name,
        "git_commit": "9780224f46",
        "git_dirty": False,
        "resumed_from": "/x/last-checkpoint" if resumed else None,
        "train_rows": 4000,
        "eval_rows": 200,
        "best_checkpoint": None if resumed else {"value": 0.0262, "step": 500},
        "wandb": {"url": f"https://wandb.example/{name}-train"},
        "gpu": "Tesla T4",
        "peak_reserved_memory_gb": 9.18,
        "wall_clock_s": 3.0 if resumed else 3500.0,
        "total_train_s": 3500.0,
        "global_step": 500,
        "trainable_params": 38_756_352,
        "cost": {"gpu_hours": 0.97, "usd": 0.34},
        "config": {
            "model": {"name": "unsloth/Qwen3.5-4B"},
            "train": {
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": 0.0002,
            },
        },
    }


def _entries(results: dict, infos: dict) -> list[dict]:
    return [
        {
            "spec": run,
            "name": f"abl_{run['name']}",
            "result": results.get(run["name"]),
            "info": infos.get(run["name"]),
        }
        for run in SPEC["runs"]
    ]


def test_report_rows_follow_the_spec_order_and_link_to_wandb():
    results = {"base": _result("abl_base", 70), "r8": _result("abl_r8", 70)}
    infos = {"base": _info("abl_base"), "r8": _info("abl_r8")}
    report = render_report(SPEC, _entries(results, infos))
    assert report.index("| `abl_base` |") < report.index("| `abl_r8` |")
    assert "| baseline |" in report
    assert "[train](https://wandb.example/abl_r8-train)" in report
    assert "[eval](https://wandb.example/abl_r8-eval)" in report
    assert "`lora.r=8`" in report
    assert "38.8 M" in report
    assert "| 7.0 |" in report  # 3500 s / 500 steps
    assert "4,000 train rows, 500 steps, batch 8" in report


def test_no_difference_says_none():
    results = {"base": _result("abl_base", 70), "r8": _result("abl_r8", 70)}
    report = render_report(SPEC, _entries(results, {}))
    assert "None. No run is different from the base run" in report


def test_a_significant_difference_is_listed():
    results = {"base": _result("abl_base", 90), "worse": _result("abl_worse", 40)}
    report = render_report(SPEC, _entries(results, {}))
    assert "`abl_worse` (worse): normalized match is 50.0 points lower" in report
    assert "`abl_worse` (worse): render match is 50.0 points lower" in report
    assert "-50.0 [" in report and "] *" in report


def test_missing_files_are_listed():
    results = {"base": _result("abl_base", 70)}
    report = render_report(SPEC, _entries(results, {"base": _info("abl_base")}))
    assert "## Runs with missing files" in report
    assert "`abl_r8`, `abl_worse`" in report


def test_resumed_runs_get_a_note():
    results = {"base": _result("abl_base", 70)}
    report = render_report(SPEC, _entries(results, {"base": _info("abl_base", resumed=True)}))
    assert "9.18†" in report
    assert "`abl_base`: the last session took 3 s of 3,500 s" in report
    assert "eval_loss is `-`" in report


def test_skipped_runs_have_their_reason_on_one_line():
    report = render_report(SPEC, _entries({"base": _result("abl_base", 70)}, {}))
    assert "| `dora` | DoRA | `lora.use_dora=true` | nan loss from step 1 |" in report


def test_write_report_reads_the_files_with_a_tag(tmp_path):
    spec_path = tmp_path / "ablations.yaml"
    spec_path.write_text(yaml.safe_dump(SPEC))
    results, runs = tmp_path / "results", tmp_path / "runs"
    results.mkdir()
    runs.mkdir()
    for name in ("base", "r8", "worse"):
        run = f"abl_{name}_smoke"
        (results / f"{run}__latex_ocr_val_n100.json").write_text(json.dumps(_result(run, 70)))
        (runs / f"{run}.json").write_text(json.dumps(_info(run)))
    out = write_report(
        spec_path, results, runs, tmp_path / "ablations.md", eval_rows=100, tag="smoke"
    )
    text = out.read_text()
    assert "| `abl_worse_smoke` |" in text
    assert "missing files" not in text
    entries = load_ablations(SPEC, results, runs, eval_rows=100, tag="smoke")
    assert all(e["result"] and e["info"] for e in entries)
