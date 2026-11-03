"""Predictors: they turn a batch of test rows into raw model outputs.

A predictor gets rows (dicts with ``image`` and ``source_index``) and returns one raw
string for each row. ``run_eval`` then applies ``extract_latex`` and the metrics, the
same way for every predictor. Heavy imports happen inside the predictor classes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from mathscribe.data.formatting import inference_messages
from mathscribe.eval.config import EvalModelConfig, GenerationConfig

log = logging.getLogger("mathscribe.eval")


class Predictor(Protocol):
    def predict(self, rows: list[dict[str, Any]]) -> list[str]: ...

    def runtime_info(self) -> dict[str, Any]: ...


class JsonlPredictor:
    """Replay the raw outputs of an earlier run (``predictions.jsonl``). No GPU needed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.raw: dict[int, str] = {}
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.raw[int(record["source_index"])] = record["raw"]

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        missing = [r["source_index"] for r in rows if r["source_index"] not in self.raw]
        if missing:
            raise KeyError(f"{self.path} has no prediction for rows {missing[:5]}")
        return [self.raw[r["source_index"]] for r in rows]

    def runtime_info(self) -> dict[str, Any]:
        """The runtime of the original run (from its ``metrics.json``), if it is there."""
        info: dict[str, Any] = {}
        original = self.path.with_name("metrics.json")
        if original.is_file():
            info.update(json.loads(original.read_text()).get("runtime", {}))
        info["replayed_from"] = str(self.path)
        return info


class UnslothPredictor:
    """Greedy generation with an Unsloth vision model (a base model or a LoRA adapter)."""

    def __init__(self, model_cfg: EvalModelConfig, gen_cfg: GenerationConfig):
        # Unsloth must be imported before transformers / trl / peft.
        import unsloth  # noqa: F401

        # isort: split
        import torch
        from unsloth import FastVisionModel

        self.torch = torch
        self.gen_cfg = gen_cfg
        self.model, self.processor = FastVisionModel.from_pretrained(
            model_cfg.name, load_in_4bit=model_cfg.load_in_4bit
        )
        FastVisionModel.for_inference(self.model)
        # Decoder-only generation needs left padding in a batch.
        getattr(self.processor, "tokenizer", self.processor).padding_side = "left"
        self.prompt = self.processor.apply_chat_template(
            inference_messages(gen_cfg.instruction),
            add_generation_prompt=True,
            **gen_cfg.chat_template_kwargs,
        )
        log.info("Prompt: %r", self.prompt)

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        images = [r["image"].convert("RGB") for r in rows]
        inputs = self.processor(
            images=images,
            text=[self.prompt] * len(images),
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_cfg.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)

    def runtime_info(self) -> dict[str, Any]:
        torch = self.torch
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu": props.name,
            "gpu_total_memory_gb": round(props.total_memory / 1024**3, 3),
            "peak_reserved_memory_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        }


def build_predictor(model_cfg: EvalModelConfig, gen_cfg: GenerationConfig) -> Predictor:
    if model_cfg.backend == "jsonl":
        return JsonlPredictor(model_cfg.name)
    if model_cfg.backend == "unsloth":
        return UnslothPredictor(model_cfg, gen_cfg)
    raise ValueError(f"Unknown backend {model_cfg.backend!r}")
