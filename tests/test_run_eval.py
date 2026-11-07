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


def _cfg(tmp_path, manifest, render=False, **model):
    return eval_config_from_dict(
        {
            "run_name": "fake",
            "model": {"label": "Fake model", **model},
            "generation": {"batch_size": 3},
            "testset": {"manifest": str(manifest)},
            "render": {"enabled": render, "cache_dir": str(tmp_path / "render_cache")},
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
    # Per-row scores in the test set order, and the intervals.
    column = "".join("1" if i % 2 == 0 else "0" for i, _ in indices)
    assert saved["per_row"]["normalized_match"] == column
    assert len(saved["per_row"]["char_edits"]) == 4
    lo, hi = saved["ci95"]["normalized_match"]
    assert lo <= saved["metrics"]["normalized_match"] <= hi
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


def test_run_with_the_renderer_adds_compile_and_render_match(tmp_path, manifest, source_ds, texts):
    from mathscribe.eval.render import Renderer, RendererUnavailable

    try:
        Renderer().close()
    except RendererUnavailable as e:
        pytest.skip(str(e))
    result = run_eval.run(_cfg(tmp_path, manifest, render=True), FakePredictor(texts), source_ds)
    m = result["metrics"]
    indices = [i for i, _ in json.loads(manifest.read_text())["rows"]]
    correct = sum(1 for i in indices if i % 2 == 0) / len(indices)
    # The fake outputs are the references (in $$...$$) or "wrong", which renders but differs.
    assert m["compile_rate"] == 1.0
    assert m["ref_render_rate"] == 1.0
    assert m["render_match"] == pytest.approx(correct)
    assert "render_match" in result["ci95"]
    assert result["renderer"].startswith("mathjax-")

    lines = (tmp_path / "out" / "fake" / "toy_n4" / "predictions.jsonl").read_text().splitlines()
    record = json.loads(lines[0])
    assert {"compiles", "compile_error", "render_match", "render_similarity"} <= set(record)
    assert any((tmp_path / "render_cache").rglob("*.png"))
    board = (tmp_path / "leaderboard.md").read_text()
    assert "Render match" in board


def test_result_records_the_scorer_version(tmp_path, manifest, source_ds, texts):
    from mathscribe.eval.metrics import SCORER_VERSION

    result = run_eval.run(_cfg(tmp_path, manifest), FakePredictor(texts), source_ds)
    assert result["scorer_version"] == SCORER_VERSION
    assert isinstance(result["git_dirty"], bool | None)
