"""Tests for the GPU-free parts of mathscribe.train.sft."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mathscribe.config import load_config
from mathscribe.train import sft

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "train" / "sft_smoke.yaml"


def _cfg(tmp_path, *overrides):
    return load_config(SMOKE, [f"train.output_dir={tmp_path}", *overrides])


def _make_ckpt(root: Path, name: str, global_step: int | None = None) -> Path:
    path = root / name
    path.mkdir(parents=True)
    if global_step is not None:
        (path / "trainer_state.json").write_text(json.dumps({"global_step": global_step}))
    return path


def test_importing_sft_does_not_load_gpu_stack():
    # A new process: other tests may import transformers (and torch) in this one.
    code = (
        "import sys, mathscribe.train.sft; "
        "assert not {'torch', 'unsloth', 'wandb'} & set(sys.modules), 'GPU stack loaded'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_build_sft_args_without_hub(tmp_path):
    args = sft.build_sft_args(_cfg(tmp_path))
    assert args["remove_unused_columns"] is False
    assert args["dataset_text_field"] == ""
    assert args["dataset_kwargs"] == {"skip_prepare_dataset": True}
    assert args["max_steps"] == 15
    assert args["run_name"] == "sft_smoke"
    assert "push_to_hub" not in args
    assert args["report_to"] == "none"


def test_build_sft_args_report_to_follows_tracking(tmp_path):
    args = sft.build_sft_args(_cfg(tmp_path, "tracking.enabled=true"))
    assert args["report_to"] == ["wandb"]


def test_best_checkpoint_info(tmp_path):
    assert sft.best_checkpoint_info(None, None, None, True) is None
    ckpt = _make_ckpt(tmp_path, "checkpoint-300", global_step=300)
    info = sft.best_checkpoint_info(str(ckpt), 0.03, "eval_loss", True)
    assert info == {
        "path": str(ckpt),
        "step": 300,
        "metric": "eval_loss",
        "value": 0.03,
        "found": True,
        "loaded": True,
    }
    # A disconnect deleted the folder: the trainer keeps the last adapter.
    gone = sft.best_checkpoint_info(str(tmp_path / "checkpoint-200"), 0.02, "loss", True)
    assert (gone["step"], gone["metric"], gone["found"], gone["loaded"]) == (
        200,
        "eval_loss",
        False,
        False,
    )


def test_build_sft_args_with_hub(tmp_path):
    cfg = _cfg(tmp_path, "hub.push=true", "hub.repo_id=me/mathscribe")
    args = sft.build_sft_args(cfg)
    assert args["push_to_hub"] is True
    assert args["hub_model_id"] == "me/mathscribe"
    assert args["hub_strategy"] == "checkpoint"
    assert args["hub_private_repo"] is True


def test_vision_args_cannot_be_overridden_by_config(tmp_path):
    cfg = _cfg(tmp_path, "train.remove_unused_columns=true")
    assert sft.build_sft_args(cfg)["remove_unused_columns"] is False


def test_find_last_local_checkpoint(tmp_path):
    assert sft.find_last_local_checkpoint(tmp_path / "missing") is None
    assert sft.find_last_local_checkpoint(tmp_path) is None

    _make_ckpt(tmp_path, "checkpoint-5")
    _make_ckpt(tmp_path, "checkpoint-10")
    _make_ckpt(tmp_path, "checkpoint-9")
    _make_ckpt(tmp_path, "not-a-checkpoint")
    # numeric order, not string order
    assert sft.find_last_local_checkpoint(tmp_path).name == "checkpoint-10"


def test_checkpoint_step_prefers_trainer_state(tmp_path):
    assert sft.checkpoint_step(_make_ckpt(tmp_path, "last-checkpoint", global_step=12)) == 12
    assert sft.checkpoint_step(_make_ckpt(tmp_path, "checkpoint-7")) == 7
    assert sft.checkpoint_step(_make_ckpt(tmp_path, "other")) == -1


def test_resume_picks_most_advanced_of_local_and_hub(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "checkpoint-5", global_step=5)
    hub_dir = tmp_path / sft.HUB_RESUME_DIR
    calls = []

    def fake_download(repo_id, dest):
        calls.append((repo_id, dest))
        return _make_ckpt(hub_dir, sft.HUB_CHECKPOINT_DIR, global_step=10)

    monkeypatch.setattr(sft, "download_hub_checkpoint", fake_download)
    cfg = _cfg(tmp_path, "hub.push=true", "hub.repo_id=me/x")

    best = sft.resolve_resume_checkpoint(cfg)
    assert best == hub_dir / sft.HUB_CHECKPOINT_DIR
    assert calls == [("me/x", hub_dir)]


def test_resume_local_only_when_hub_is_off(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "checkpoint-5", global_step=5)
    monkeypatch.setattr(
        sft, "download_hub_checkpoint", lambda *a: pytest.fail("Hub must not be called")
    )
    assert sft.resolve_resume_checkpoint(_cfg(tmp_path)).name == "checkpoint-5"


def test_resume_without_any_checkpoint_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(sft, "download_hub_checkpoint", lambda *a: None)
    cfg = _cfg(tmp_path, "hub.push=true", "hub.repo_id=me/x")
    with pytest.raises(SystemExit, match="no checkpoint"):
        sft.resolve_resume_checkpoint(cfg)


def test_hub_resume_dir_is_not_pushed_by_trainer():
    # transformers skips paths that start with "_" when it pushes output_dir.
    assert sft.HUB_RESUME_DIR.startswith("_")


def test_dry_run_prints_resolved_config(capsys):
    sft.main(["--config", str(SMOKE), "--set", "train.max_steps=2", "--dry-run"])
    out = json.loads(capsys.readouterr().out)
    assert out["sft_args"]["max_steps"] == 2
    assert out["config"]["model"]["name"] == "unsloth/Qwen3.5-4B"


def test_resume_if_available_starts_a_new_run_without_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sft, "download_hub_checkpoint", lambda *a: None)
    cfg = _cfg(tmp_path, "hub.push=true", "hub.repo_id=me/x")
    assert sft.resolve_resume_checkpoint(cfg, required=False) is None


def test_resume_if_available_uses_a_checkpoint_when_there_is_one(tmp_path, monkeypatch):
    _make_ckpt(tmp_path, "checkpoint-5", global_step=5)
    monkeypatch.setattr(sft, "download_hub_checkpoint", lambda *a: None)
    cfg = _cfg(tmp_path, "hub.push=true", "hub.repo_id=me/x")
    assert sft.resolve_resume_checkpoint(cfg, required=False).name == "checkpoint-5"


def test_resume_flags_exclude_each_other():
    with pytest.raises(SystemExit, match="not both"):
        sft.main(["--config", "configs/train/sft_smoke.yaml", "--resume", "--resume-if-available"])
