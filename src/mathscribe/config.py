"""Run configuration: YAML file + ``key.path=value`` overrides -> typed dataclasses."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mathscribe.data.formatting import INSTRUCTION


class ConfigError(ValueError):
    """The config file or an override is not valid."""


@dataclass
class ModelConfig:
    name: str = "unsloth/Qwen3.5-4B"
    load_in_4bit: bool = False
    use_gradient_checkpointing: str | bool = "unsloth"


@dataclass
class LoraConfig:
    finetune_vision_layers: bool = True
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    bias: str = "none"
    use_rslora: bool = False
    random_state: int = 2003


@dataclass
class DataConfig:
    dataset: str = "unsloth/LaTeX_OCR"
    split: str = "train"
    max_train_samples: int | None = None
    instruction: str = INSTRUCTION


@dataclass
class HubConfig:
    """Checkpoint push to the HF Hub. ``repo_id`` is required when ``push`` is true."""

    push: bool = False
    repo_id: str | None = None
    private: bool = True


@dataclass
class SFTRunConfig:
    run_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    # Passed as-is to trl.SFTConfig (see mathscribe.train.sft.build_sft_args).
    train: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.train["output_dir"])

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {"model": ModelConfig, "lora": LoraConfig, "data": DataConfig, "hub": HubConfig}


def parse_override(override: str) -> tuple[list[str], Any]:
    """Parse ``a.b.c=value``. The value is read as YAML, so ``5``, ``true``, ``null`` work."""
    key, sep, raw = override.partition("=")
    key = key.strip()
    if not sep or not key or any(not part for part in key.split(".")):
        raise ConfigError(f"Override must look like 'section.key=value', got {override!r}")
    return key.split("."), yaml.safe_load(raw)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    for override in overrides:
        path, value = parse_override(override)
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Cannot set {override!r}: {part!r} is not a section")
        node[path[-1]] = value
    return out


def _build_section(cls: type, name: str, values: Any) -> Any:
    values = values or {}
    if not isinstance(values, dict):
        raise ConfigError(f"Section {name!r} must be a mapping")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown keys in {name!r}: {sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> SFTRunConfig:
    allowed = {"run_name", "train", *_SECTIONS}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {sorted(unknown)}")
    if not raw.get("run_name"):
        raise ConfigError("'run_name' is required")

    train = raw.get("train") or {}
    if not isinstance(train, dict) or not train.get("output_dir"):
        raise ConfigError("'train.output_dir' is required")

    cfg = SFTRunConfig(
        run_name=raw["run_name"],
        train=dict(train),
        **{name: _build_section(cls, name, raw.get(name)) for name, cls in _SECTIONS.items()},
    )
    if cfg.hub.push and not cfg.hub.repo_id:
        raise ConfigError("'hub.repo_id' is required when 'hub.push' is true")
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> SFTRunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(apply_overrides(raw, overrides or []))
