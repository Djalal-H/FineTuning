import json
from pathlib import Path

from mathscribe.config import load_config
from mathscribe.train import model_card

BASE = Path(__file__).resolve().parents[1] / "configs" / "train" / "sft_base.yaml"


def _info(**extra):
    cfg = load_config(BASE)
    return {
        "run_name": "sft_base",
        "git_commit": "0123456789abcdef",
        "git_remote": "https://github.com/me/mathscribe",
        "train_rows": 12000,
        "global_step": 1500,
        "trainable_params": 38756352,
        "total_params": 4578021888,
        "gpu": "Tesla T4",
        "wall_clock_s": 3600.0,
        "total_train_s": 25200.0,
        "peak_reserved_memory_gb": 9.7,
        "cost": {"gpu_hours": 7.0, "usd_per_hour": 0.35, "usd": 2.45},
        "best_checkpoint": {"step": 1200, "metric": "eval_loss", "value": 0.0312},
        "wandb": {"id": "abc123", "url": "https://wandb.ai/me/mathscribe/runs/abc123"},
        "config": cfg.to_dict(),
        **extra,
    }


def _result(run_name, testset, n=1000):
    return {
        "run_name": run_name,
        "git_commit": "fedcba9876543210",
        "git_dirty": False,
        "testset": {"name": testset, "n": n},
        "metrics": {"exact_match": 0.5, "normalized_match": 0.6, "cer": 0.1},
        "ci95": {"normalized_match": [0.57, 0.63]},
    }


def test_card_without_eval_results():
    card = model_card.build_model_card(_info(), [], "me/mathscribe-sft-base")
    assert card.startswith("---\nbase_model: unsloth/Qwen3.5-4B\n")
    assert "No eval results yet" in card
    assert "12000 rows" in card
    assert "excluded manifests: `latex_ocr_val.json`" in card
    assert "Best eval_loss: 0.0312 (step 1200)" in card
    assert "$2.45 at $0.35/GPU-hour" in card
    assert "[abc123](https://wandb.ai/me/mathscribe/runs/abc123)" in card
    assert "enable_thinking=False" in card
    assert 'from_pretrained("me/mathscribe-sft-base"' in card
    assert "[MathScribe](https://github.com/me/mathscribe) (commit `0123456`)" in card


def test_card_with_eval_results():
    results = [_result("sft_base", "latex_ocr_test")]
    card = model_card.build_model_card(_info(cost={"gpu_hours": 0.5}), results)
    row = "| `latex_ocr_test` | 1000 | 50.0 | 60.0 [57.0, 63.0] | 10.0 | - | - | `fedcba9` |"
    assert row in card
    assert "no public price" in card


def test_eval_results_for_filters_by_run_name(tmp_path):
    for name, r in {
        "a": _result("sft_base", "latex_ocr_test"),
        "b": _result("sft_base", "crohme2014", 986),
        "c": _result("qwen35_4b_zeroshot", "latex_ocr_test"),
    }.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(r))
    found = model_card.eval_results_for("sft_base", tmp_path)
    assert [r["testset"]["name"] for r in found] == ["crohme2014", "latex_ocr_test"]
    assert model_card.eval_results_for("sft_base", tmp_path / "missing") == []


def test_cli_writes_card_from_local_run_info(tmp_path):
    run_info = tmp_path / "run_info.json"
    run_info.write_text(json.dumps(_info()))
    model_card.main(["--run-info", str(run_info), "--results-dir", str(tmp_path / "none")])
    assert (tmp_path / "README.md").read_text().startswith("---\n")
