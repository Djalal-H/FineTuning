import dataclasses
from pathlib import Path

import pytest
import yaml

from mathscribe.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ABLATIONS = yaml.safe_load((ROOT / "configs" / "ablations.yaml").read_text())
BASE = ROOT / ABLATIONS["base_config"]
RUNS = ABLATIONS["runs"]


def _flat(prefix: str, obj) -> dict:
    if dataclasses.is_dataclass(obj):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            out.update(_flat(f"{prefix}.{key}" if prefix else key, value))
        return out
    return {prefix: obj}


def _differences(a, b) -> set[str]:
    flat_a, flat_b = _flat("", a), _flat("", b)
    return {key for key in flat_a if flat_a[key] != flat_b.get(key)}


def test_run_names_are_unique():
    names = [run["name"] for run in RUNS]
    assert len(names) == len(set(names))
    assert names[0] == "base"


@pytest.mark.parametrize("run", RUNS, ids=lambda r: r["name"])
def test_run_loads_and_changes_only_its_factor(run):
    base = load_config(BASE)
    cfg = load_config(BASE, run["overrides"])
    changed = _differences(cfg, base)
    stated = {o.partition("=")[0] for o in run["overrides"]}
    # An override that sets the base value again changes nothing, so "changed" can be smaller.
    assert changed <= stated
    assert len(stated) == len(run["overrides"])  # no key twice
    if run["name"] != "base":
        assert changed  # the run must differ from the base


def test_base_run_has_no_overrides():
    assert RUNS[0]["overrides"] == []


def test_ablation_base_is_the_fixed_subset():
    cfg = load_config(BASE)
    assert cfg.data.max_train_samples == 4000
    assert cfg.validation.manifest in cfg.data.exclude
    assert cfg.validation.early_stopping_patience is None  # every run trains all steps
    assert cfg.train["load_best_model_at_end"] is True


def test_dora_is_a_config_key():
    assert load_config(BASE, ["lora.use_dora=true"]).lora.use_dora is True
    assert load_config(BASE).lora.use_dora is False
