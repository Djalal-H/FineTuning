"""Eval run configuration: YAML file + ``key.path=value`` overrides -> typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mathscribe.config import ConfigError, apply_overrides, build_section
from mathscribe.data.formatting import CHAT_TEMPLATE_KWARGS, INSTRUCTION

BACKENDS = ("unsloth", "anthropic", "pix2tex", "got_ocr", "jsonl")


@dataclass
class EvalModelConfig:
    """The model to evaluate.

    - ``backend: unsloth``: ``name`` is a base model, or a LoRA adapter (Hub repo or
      local folder). Unsloth reads the base model from the adapter config.
    - ``backend: anthropic``: ``name`` is a Claude model ID.
    - ``backend: pix2tex``: ``name`` is only a label (pix2tex has one checkpoint).
    - ``backend: got_ocr``: ``name`` is a GOT-OCR 2.0 checkpoint on the Hub.
    - ``backend: jsonl``: ``name`` is a ``predictions.jsonl`` file of an earlier run.
      The run scores the stored raw outputs again, with no model and no GPU.
    """

    backend: str = "unsloth"
    name: str | None = "unsloth/Qwen3.5-4B"
    label: str | None = None  # name in the leaderboard (default: run_name)
    load_in_4bit: bool = False
    options: dict[str, Any] = field(default_factory=dict)  # backend-specific, see predictors.py


@dataclass
class GenerationConfig:
    instruction: str = INSTRUCTION
    system_prompt: str | None = None  # only the anthropic backend uses it
    max_new_tokens: int = 512
    batch_size: int = 8
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: dict(CHAT_TEMPLATE_KWARGS))


@dataclass
class TestsetConfig:
    manifest: str = "data/testsets/latex_ocr_test.json"
    subset: int | None = None  # the first N rows of the manifest; None = all rows


@dataclass
class RenderConfig:
    """Compile rate and render match (see ``mathscribe.eval.render``). Needs node."""

    enabled: bool = True
    timeout_s: float = 10.0
    cache_dir: str | None = "outputs/render_cache"


@dataclass
class OutputConfig:
    dir: str = "outputs/eval"  # predictions.jsonl and metrics.json for each run
    results_dir: str = "reports/results"  # one small JSON per run, read by the leaderboard
    leaderboard: str = "reports/leaderboard.md"
    # The leaderboard shows each run's difference to this run (paired 95% interval).
    baseline_run: str | None = "qwen35_4b_zeroshot"


@dataclass
class EvalRunConfig:
    run_name: str
    model: EvalModelConfig = field(default_factory=EvalModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    testset: TestsetConfig = field(default_factory=TestsetConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def label(self) -> str:
        return self.model.label or self.run_name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "model": EvalModelConfig,
    "generation": GenerationConfig,
    "testset": TestsetConfig,
    "render": RenderConfig,
    "output": OutputConfig,
}


def eval_config_from_dict(raw: dict[str, Any]) -> EvalRunConfig:
    unknown = set(raw) - {"run_name", *_SECTIONS}
    if unknown:
        raise ConfigError(f"Unknown top-level keys: {sorted(unknown)}")
    if not raw.get("run_name"):
        raise ConfigError("'run_name' is required")
    cfg = EvalRunConfig(
        run_name=raw["run_name"],
        **{name: build_section(cls, name, raw.get(name)) for name, cls in _SECTIONS.items()},
    )
    if not cfg.model.name:
        raise ConfigError("'model.name' is required (set it with --set model.name=...)")
    if cfg.model.backend not in BACKENDS:
        raise ConfigError(f"model.backend must be one of {BACKENDS}, got {cfg.model.backend!r}")
    if cfg.generation.batch_size < 1:
        raise ConfigError("generation.batch_size must be >= 1")
    if cfg.testset.subset is not None and cfg.testset.subset < 1:
        raise ConfigError("testset.subset must be >= 1 or null")
    return cfg


def load_eval_config(path: str | Path, overrides: list[str] | None = None) -> EvalRunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return eval_config_from_dict(apply_overrides(raw, overrides or []))
