"""Dataset loading."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from mathscribe.data.formatting import INSTRUCTION, convert_to_conversation

if TYPE_CHECKING:
    from datasets import Dataset

    from mathscribe.config import DataConfig


def to_conversation_batch(
    batch: dict[str, list[Any]], instruction: str = INSTRUCTION
) -> dict[str, list[Any]]:
    """Batch transform: ``{"image": [...], "text": [...]}`` -> ``{"messages": [...]}``."""
    return {
        "messages": [
            convert_to_conversation({"image": image, "text": text}, instruction)["messages"]
            for image, text in zip(batch["image"], batch["text"], strict=True)
        ]
    }


def as_conversations(ds: Dataset, instruction: str = INSTRUCTION) -> Dataset:
    """Format the rows as conversations when they are read.

    The notebook converted all 68k rows (and decoded every image) up front, into a list.
    TRL >= 1.x accepts only a ``datasets.Dataset``, so this uses ``with_transform``.
    The result is a ``Dataset`` with the same rows in the same order, so the trainer's
    seeded sampler picks the same rows as the notebook. Each item is ``{"messages": ...}``.
    """
    return ds.with_transform(partial(to_conversation_batch, instruction=instruction))


def select_training_rows(ds: Dataset, cfg: DataConfig) -> Dataset:
    """Remove the rows of the ``cfg.exclude`` manifests (e.g. the val set)."""
    if not cfg.exclude:
        return ds
    from mathscribe.eval.testsets import Manifest, excluded_complement

    manifests = [Manifest.load(path) for path in cfg.exclude]
    keep = excluded_complement(len(ds), cfg.dataset, cfg.split, cfg.revision, manifests)
    return ds.select(keep)


def load_latex_ocr(cfg: DataConfig) -> Dataset:
    """Load the configured split of ``unsloth/LaTeX_OCR`` (or a compatible dataset)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, split=cfg.split, revision=cfg.revision)
    ds = select_training_rows(ds, cfg)
    if cfg.max_train_samples is not None:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))
    return as_conversations(ds, cfg.instruction)
