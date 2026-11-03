"""Dataset loading."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mathscribe.data.formatting import INSTRUCTION, convert_to_conversation

if TYPE_CHECKING:
    from mathscribe.config import DataConfig


class ConversationDataset:
    """Map-style dataset that formats each row when it is read.

    The notebook converted all 68k rows (and decoded every image) up front.
    This class gives the same items in the same order, with no up-front cost.
    The length is the same, so the trainer's seeded sampler picks the same rows.
    """

    def __init__(self, rows: Sequence[Any], instruction: str = INSTRUCTION) -> None:
        self.rows = rows
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return convert_to_conversation(self.rows[idx], self.instruction)


def load_latex_ocr(cfg: DataConfig) -> ConversationDataset:
    """Load the configured split of ``unsloth/LaTeX_OCR`` (or a compatible dataset)."""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.max_train_samples is not None:
        ds = ds.select(range(min(cfg.max_train_samples, len(ds))))
    return ConversationDataset(ds, cfg.instruction)
