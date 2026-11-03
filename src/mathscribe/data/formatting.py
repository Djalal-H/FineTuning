"""Chat formatting for vision SFT (ported from notebooks/Qwen_3_5_SFT.ipynb)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

INSTRUCTION = "Write the LaTeX representation for this image."


def convert_to_conversation(
    sample: Mapping[str, Any], instruction: str = INSTRUCTION
) -> dict[str, list[dict[str, Any]]]:
    """Turn one ``{"image", "text"}`` row into a user/assistant conversation.

    The content order (text, then image) matches the notebook training run.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
    }
