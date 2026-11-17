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
    revision: str | None = None  # dataset commit sha; None = the main branch
    # Manifests (data/testsets/*.json) whose rows training must not see, e.g. the val set.
    exclude: list[str] = field(default_factory=list)
    max_train_samples: int | None = None
    instruction: str = INSTRUCTION


@dataclass
class HubConfig:
    """Checkpoint push to the HF Hub. ``repo_id`` is required when ``push`` is true."""

    push: bool = False
    repo_id: str | None = None
    private: bool = True


@dataclass
class TrackingConfig:
    """W&B logging (see ``mathscribe.tracking``). Training and evaluation use the same section.

    This section is the only switch: ``train.report_to`` is set from ``enabled``.
    """

    enabled: bool = False
    project: str = "mathscribe"
    entity: str | None = None  # W&B team or user; None = the default entity of the API key
    group: str | None = None  # None = the run name (training) or the model run name (eval)
    tags: list[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """The validation set for ``eval_loss`` during training and for early stopping.

    ``manifest`` must also be in ``data.exclude``, so that training never sees these rows.
    """

    manifest: str | None = None  # e.g. data/testsets/latex_ocr_val.json; None = no validation
    samples: int | None = None  # the first N rows of the manifest; None = all rows
    early_stopping_patience: int | None = None  # evaluations without improvement; None = off
    early_stopping_threshold: float = 0.0


@dataclass
class SFTRunConfig:
    run_name: str
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    # Passed as-is to trl.SFTConfig (see mathscribe.train.sft.build_sft_args).
    train: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        return Path(self.train["output_dir"])

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "lora": LoraConfig,
    "data": DataConfig,
    "validation": ValidationConfig,
    "hub": HubConfig,
    "tracking": TrackingConfig,
}


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


def build_section(cls: type, name: str, values: Any) -> Any:
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
        **{name: build_section(cls, name, raw.get(name)) for name, cls in _SECTIONS.items()},
    )
    if cfg.hub.push and not cfg.hub.repo_id:
        raise ConfigError("'hub.repo_id' is required when 'hub.push' is true")
    if "report_to" in cfg.train:
        raise ConfigError("Do not set 'train.report_to'. Use 'tracking.enabled' instead")
    check_validation(cfg)
    return cfg


def check_validation(cfg: SFTRunConfig) -> None:
    """The validation settings and the ``train`` evaluation arguments must agree."""
    val, train = cfg.validation, cfg.train
    # YAML reads a bare `no` as False, so both forms mean "no evaluation".
    evaluates = train.get("eval_strategy", "no") not in ("no", False, None)
    if val.manifest is None:
        if evaluates:
            raise ConfigError("'train.eval_strategy' needs 'validation.manifest'")
        if val.early_stopping_patience is not None:
            raise ConfigError("'validation.early_stopping_patience' needs 'validation.manifest'")
        return
    if not evaluates:
        raise ConfigError("'validation.manifest' needs 'train.eval_strategy' (e.g. steps)")
    if Path(val.manifest) not in {Path(p) for p in cfg.data.exclude}:
        raise ConfigError(
            f"'validation.manifest' ({val.manifest}) must also be in 'data.exclude', "
            "or training sees the validation rows"
        )
    if val.samples is not None and val.samples < 1:
        raise ConfigError("'validation.samples' must be >= 1 or null")
    if val.early_stopping_patience is not None:
        if val.early_stopping_patience < 1:
            raise ConfigError("'validation.early_stopping_patience' must be >= 1 or null")
        if not train.get("load_best_model_at_end"):
            raise ConfigError("Early stopping needs 'train.load_best_model_at_end: true'")


def load_config(path: str | Path, overrides: list[str] | None = None) -> SFTRunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(apply_overrides(raw, overrides or []))
