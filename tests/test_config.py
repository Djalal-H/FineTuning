from pathlib import Path

import pytest

from mathscribe.config import (
    ConfigError,
    apply_overrides,
    config_from_dict,
    load_config,
    parse_override,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "train"
SMOKE = CONFIG_DIR / "sft_smoke.yaml"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_all_train_configs_load(path):
    load_config(path)


def test_smoke_config_matches_notebook():
    cfg = load_config(SMOKE)
    assert cfg.model.name == "unsloth/Qwen3.5-4B"
    assert cfg.model.load_in_4bit is False
    assert (cfg.lora.r, cfg.lora.lora_alpha, cfg.lora.random_state) == (16, 16, 2003)
    t = cfg.train
    assert t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] == 8
    assert t["max_steps"] == 15
    assert t["learning_rate"] == pytest.approx(2e-4)
    assert isinstance(t["learning_rate"], float)
    assert t["seed"] == 3407
    assert t["optim"] == "adamw_8bit"
    assert cfg.hub.push is False


@pytest.mark.parametrize(
    ("override", "path", "value"),
    [
        ("train.max_steps=30", ["train", "max_steps"], 30),
        ("hub.push=true", ["hub", "push"], True),
        ("data.max_train_samples=null", ["data", "max_train_samples"], None),
        ("hub.repo_id=user/repo", ["hub", "repo_id"], "user/repo"),
        ("train.learning_rate=1.0e-5", ["train", "learning_rate"], 1e-5),
    ],
)
def test_parse_override(override, path, value):
    assert parse_override(override) == (path, value)


@pytest.mark.parametrize("bad", ["train.max_steps", "=5", "train..x=1", ""])
def test_parse_override_rejects_bad_input(bad):
    with pytest.raises(ConfigError):
        parse_override(bad)


def test_apply_overrides_does_not_mutate_input():
    raw = {"train": {"max_steps": 15}}
    out = apply_overrides(raw, ["train.max_steps=5", "train.new_key=1"])
    assert out["train"] == {"max_steps": 5, "new_key": 1}
    assert raw == {"train": {"max_steps": 15}}


def test_load_config_with_overrides():
    cfg = load_config(SMOKE, ["train.max_steps=3", "hub.push=true", "hub.repo_id=me/x", "lora.r=8"])
    assert cfg.train["max_steps"] == 3
    assert cfg.hub.repo_id == "me/x"
    assert cfg.lora.r == 8


def _minimal(**extra):
    return {"run_name": "r", "train": {"output_dir": "out"}, **extra}


def test_unknown_section_key_is_an_error():
    with pytest.raises(ConfigError, match="Unknown keys in 'lora'"):
        config_from_dict(_minimal(lora={"rank": 8}))


def test_unknown_top_level_key_is_an_error():
    with pytest.raises(ConfigError, match="Unknown top-level keys"):
        config_from_dict(_minimal(optimizer={}))


def test_hub_push_requires_repo_id():
    with pytest.raises(ConfigError, match=r"hub\.repo_id"):
        config_from_dict(_minimal(hub={"push": True}))


def test_output_dir_is_required():
    with pytest.raises(ConfigError, match="output_dir"):
        config_from_dict({"run_name": "r", "train": {}})
