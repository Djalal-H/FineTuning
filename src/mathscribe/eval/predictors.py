"""Predictors: they turn a batch of test rows into raw model outputs.

A predictor gets rows (dicts with ``image`` and ``source_index``) and returns one raw
string for each row. ``run_eval`` then applies ``extract_latex`` and the metrics, the
same way for every predictor. Heavy imports happen inside the predictor classes.

Backends (``model.backend``):

- ``unsloth``: a Qwen-VL base model or LoRA adapter (GPU, the ``train`` extra).
- ``anthropic``: a Claude model through the API (the ``api`` extra).
- ``pix2tex``: pix2tex / LaTeX-OCR (the ``baselines`` extra).
- ``got_ocr``: GOT-OCR 2.0 through transformers (the ``baselines`` extra).
- ``jsonl``: replay the raw outputs of an earlier run.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from collections import Counter
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


def _png_base64(image: Any) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


class AnthropicPredictor:
    """A Claude model through the Messages API (``pip install -e ".[api]"``, no GPU).

    The user turn has the instruction, then the image (the training order). The
    ``generation.system_prompt`` asks for the LaTeX only. ``generation.batch_size`` is the
    number of parallel requests. Options (``model.options``):

    - ``effort`` (default ``low``): ``output_config.effort``. Claude Opus 5.5 always thinks,
      and it has no ``temperature``; a low effort keeps the answers short.
    - ``fallbacks`` (default true): server-side refusal fallbacks. A row that a fallback
      model answers is counted in ``runtime_info()["fallback_rows"]``.
    - ``max_tokens`` (default 4096): thinking and the answer together.
    """

    FALLBACK_BETA = "server-side-fallback-2026-07-01"

    def __init__(self, model_cfg: EvalModelConfig, gen_cfg: GenerationConfig):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=6)
        self.model = model_cfg.name
        self.gen_cfg = gen_cfg
        self.effort = model_cfg.options.get("effort", "low")
        self.fallbacks = bool(model_cfg.options.get("fallbacks", True))
        self.max_tokens = int(model_cfg.options.get("max_tokens", 4096))
        self._lock = threading.Lock()
        self.stats: Counter[str] = Counter()

    def _one(self, image: Any) -> str:
        content = [
            {"type": "text", "text": self.gen_cfg.instruction},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": _png_base64(image)},
            },
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "output_config": {"effort": self.effort},
            "messages": [{"role": "user", "content": content}],
        }
        if self.gen_cfg.system_prompt:
            kwargs["system"] = self.gen_cfg.system_prompt
        if self.fallbacks:
            response = self.client.beta.messages.create(
                **kwargs, betas=[self.FALLBACK_BETA], fallbacks="default"
            )
        else:
            response = self.client.messages.create(**kwargs)
        iterations = getattr(response.usage, "iterations", None) or []
        with self._lock:
            self.stats["requests"] += 1
            self.stats["input_tokens"] += response.usage.input_tokens
            self.stats["output_tokens"] += response.usage.output_tokens
            self.stats["refusals"] += response.stop_reason == "refusal"
            self.stats["max_tokens_stops"] += response.stop_reason == "max_tokens"
            self.stats["fallback_rows"] += any(
                getattr(it, "type", None) == "fallback_message" for it in iterations
            )
        return "".join(block.text for block in response.content if block.type == "text")

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.gen_cfg.batch_size) as pool:
            return list(pool.map(self._one, [r["image"] for r in rows]))

    def runtime_info(self) -> dict[str, Any]:
        return {"gpu": "API", "api_model": self.model, "effort": self.effort, **self.stats}


class Pix2TexPredictor:
    """pix2tex / LaTeX-OCR (``pip install -e ".[baselines]"``). It runs one image at a time.

    ``model.name`` is ignored: pix2tex downloads its own default checkpoint.
    """

    def __init__(self, model_cfg: EvalModelConfig, gen_cfg: GenerationConfig):
        import torch
        from munch import Munch
        from pix2tex.cli import LatexOCR

        self.torch = torch
        # LatexOCR() with no arguments runs on the CPU (no_cuda: True). It also samples
        # with its config temperature (0.2), so a near-zero temperature makes it greedy.
        # It sets the root logger to FATAL; run_eval's own logger keeps its INFO level.
        self.model = LatexOCR(
            Munch(
                {
                    "config": "settings/config.yaml",
                    "checkpoint": "checkpoints/weights.pth",
                    "no_cuda": not torch.cuda.is_available(),
                    "no_resize": False,
                    "temperature": 1e-5,
                }
            )
        )

        self.failed_rows: list[int] = []

    def _one(self, row: dict[str, Any]) -> str:
        try:
            return self.model(row["image"].convert("RGB"))
        except Exception as e:  # pix2tex's own resizer can make an empty image
            log.warning("pix2tex failed on row %s: %s", row["source_index"], str(e)[:200])
            self.failed_rows.append(row["source_index"])
            return ""  # an empty prediction: wrong in every metric

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        return [self._one(r) for r in rows]

    def runtime_info(self) -> dict[str, Any]:
        return {**_torch_runtime(self.torch), "failed_rows": self.failed_rows}


class GotOcrPredictor:
    """GOT-OCR 2.0 through transformers, with ``format=True`` (LaTeX for math).

    ``pip install -e ".[baselines]"``. ``model.name`` is a GOT-OCR checkpoint, for example
    ``stepfun-ai/GOT-OCR-2.0-hf``.
    """

    def __init__(self, model_cfg: EvalModelConfig, gen_cfg: GenerationConfig):
        import torch
        from transformers import AutoProcessor, GotOcr2ForConditionalGeneration

        self.torch = torch
        self.gen_cfg = gen_cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_cfg.name)
        self.model = GotOcr2ForConditionalGeneration.from_pretrained(
            model_cfg.name, torch_dtype=dtype, device_map=device
        )

    def predict(self, rows: list[dict[str, Any]]) -> list[str]:
        images = [r["image"].convert("RGB") for r in rows]
        inputs = self.processor(images, return_tensors="pt", format=True).to(
            self.model.device, dtype=self.model.dtype
        )
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self.processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=self.gen_cfg.max_new_tokens,
            )
        generated = out[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)

    def runtime_info(self) -> dict[str, Any]:
        return _torch_runtime(self.torch)


def _torch_runtime(torch: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"gpu": "CPU"}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu": props.name,
        "gpu_total_memory_gb": round(props.total_memory / 1024**3, 3),
        "peak_reserved_memory_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


PREDICTORS: dict[str, Any] = {
    "unsloth": UnslothPredictor,
    "anthropic": AnthropicPredictor,
    "pix2tex": Pix2TexPredictor,
    "got_ocr": GotOcrPredictor,
}


def build_predictor(model_cfg: EvalModelConfig, gen_cfg: GenerationConfig) -> Predictor:
    if model_cfg.backend == "jsonl":
        return JsonlPredictor(model_cfg.name)
    if model_cfg.backend in PREDICTORS:
        return PREDICTORS[model_cfg.backend](model_cfg, gen_cfg)
    raise ValueError(f"Unknown backend {model_cfg.backend!r}")
