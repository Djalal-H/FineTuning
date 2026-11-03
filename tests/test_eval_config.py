from pathlib import Path

import pytest

from mathscribe.config import ConfigError
from mathscribe.data import CHAT_TEMPLATE_KWARGS, INSTRUCTION
from mathscribe.eval.config import eval_config_from_dict, load_eval_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_all_eval_configs_load(path):
    cfg = load_eval_config(path, ["model.name=me/adapter"])
    # Every eval uses the training prompt and turns thinking off.
    assert cfg.generation.instruction == INSTRUCTION
    assert cfg.generation.chat_template_kwargs == CHAT_TEMPLATE_KWARGS
    assert Path(cfg.testset.manifest).name == "latex_ocr_test.json"


def test_sft_smoke_needs_a_model_name():
    with pytest.raises(ConfigError, match=r"model\.name"):
        load_eval_config(CONFIG_DIR / "sft_smoke.yaml")


def test_defaults_and_label():
    cfg = eval_config_from_dict({"run_name": "r"})
    assert cfg.label == "r"
    assert cfg.model.backend == "unsloth"
    assert cfg.generation.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({}, "run_name"),
        ({"run_name": "r", "extra": 1}, "Unknown top-level"),
        ({"run_name": "r", "model": {"backend": "nope"}}, "backend"),
        ({"run_name": "r", "model": {"typo": 1}}, "Unknown keys"),
        ({"run_name": "r", "generation": {"batch_size": 0}}, "batch_size"),
        ({"run_name": "r", "testset": {"subset": 0}}, "subset"),
    ],
)
def test_invalid_configs(raw, match):
    with pytest.raises(ConfigError, match=match):
        eval_config_from_dict(raw)
