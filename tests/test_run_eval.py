"""End-to-end run_eval with a fake predictor and a local dataset (no GPU, no Hub)."""

import json
import sys
from pathlib import Path

import pytest

from mathscribe.eval import run_eval
from mathscribe.eval.config import eval_config_from_dict

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"


class FakePredictor:
    """Correct (with a math wrapper) on even source rows, wrong on odd rows."""

    def __init__(self, texts):
        self.texts = texts
        self.batches = []

    def predict(self, rows):
        self.batches.append(len(rows))
        return [
            f"$${self.texts[r['source_index']]}$$" if r["source_index"] % 2 == 0 else "wrong"
            for r in rows
        ]

    def runtime_info(self):
        return {"gpu": "FakeGPU"}


def _cfg(tmp_path, manifest, **model):
    return eval_config_from_dict(
        {
            "run_name": "fake",
            "model": {"label": "Fake model", **model},
            "generation": {"batch_size": 3},
            "testset": {"manifest": str(manifest)},
            "output": {
                "dir": str(tmp_path / "out"),
                "results_dir": str(tmp_path / "results"),
                "leaderboard": str(tmp_path / "leaderboard.md"),
            },
        }
    )


def test_importing_run_eval_does_not_load_gpu_stack():
    assert "torch" not in sys.modules
    assert "unsloth" not in sys.modules


def test_run_writes_predictions_result_and_leaderboard(tmp_path, manifest, source_ds, texts):
    cfg = _cfg(tmp_path, manifest)
    predictor = FakePredictor(texts)
    result = run_eval.run(cfg, predictor=predictor, source=source_ds)

    assert predictor.batches == [3, 1]
    indices = json.loads(manifest.read_text())["rows"]
    expected = sum(1 for i, _ in indices if i % 2 == 0) / len(indices)
    assert result["metrics"]["n"] == 4
    assert result["metrics"]["normalized_match"] == pytest.approx(expected)
    assert result["runtime"]["gpu"] == "FakeGPU"
    assert result["testset"]["n"] == 4

    run_dir = tmp_path / "out" / "fake" / "toy_n4"
    records = [
        json.loads(line) for line in (run_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert [r["source_index"] for r in records] == [i for i, _ in indices]
    first = records[0]
    assert first["reference"] == texts[first["source_index"]]
    if first["source_index"] % 2 == 0:
        assert first["raw"].startswith("$$")
        assert first["prediction"] == first["reference"]

    saved = json.loads((tmp_path / "results" / "fake__toy_n4.json").read_text())
    assert saved["metrics"] == result["metrics"]
    board = (tmp_path / "leaderboard.md").read_text()
    assert "## toy, n=4" in board
    assert "| Fake model |" in board


def test_subset_uses_the_first_rows_and_its_own_hash(tmp_path, manifest, source_ds, texts):
    full = run_eval.run(_cfg(tmp_path, manifest), FakePredictor(texts), source_ds)
    cfg = _cfg(tmp_path, manifest)
    cfg.testset.subset = 2
    small = run_eval.run(cfg, FakePredictor(texts), source_ds)
    assert small["testset"]["n"] == 2
    assert small["testset"]["sha256"] != full["testset"]["sha256"]
    assert (tmp_path / "results" / "fake__toy_n2.json").is_file()


def test_jsonl_backend_replays_raw_outputs(tmp_path, manifest, source_ds, texts):
    first = run_eval.run(_cfg(tmp_path, manifest), FakePredictor(texts), source_ds)
    predictions = tmp_path / "out" / "fake" / "toy_n4" / "predictions.jsonl"

    from mathscribe.eval.predictors import build_predictor

    cfg = _cfg(tmp_path, manifest, backend="jsonl", name=str(predictions))
    replay = run_eval.run(cfg, build_predictor(cfg.model, cfg.generation), source_ds)
    assert replay["metrics"] == first["metrics"]
    # The replay keeps the runtime of the original run.
    assert replay["runtime"]["gpu"] == "FakeGPU"
    assert replay["runtime"]["replayed_from"] == str(predictions)


def test_cli_dry_run(capsys):
    run_eval.main(
        ["--config", str(CONFIG_DIR / "qwen35_4b_zeroshot.yaml"), "--subset", "10", "--dry-run"]
    )
    out = json.loads(capsys.readouterr().out)
    assert out["testset"]["subset"] == 10


def test_cli_reports_config_errors():
    with pytest.raises(SystemExit, match=r"model\.name"):
        run_eval.main(["--config", str(CONFIG_DIR / "sft_smoke.yaml"), "--dry-run"])
