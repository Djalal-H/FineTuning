"""Chat formatting for vision SFT and for inference (ported from notebooks/Qwen_3_5_SFT.ipynb).

Training and inference build the user turn with the same function, ``user_content``.
The order is text, then image (the order of the notebook training run). The notebook
used image-then-text at inference time; this module does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

INSTRUCTION = "Write the LaTeX representation for this image."

# Keyword arguments for ``apply_chat_template`` at inference time.
# The Qwen3.5 template renders the last assistant turn of a training conversation as
# "<think>\n\n</think>\n\n" + answer. With add_generation_prompt=True it appends only
# "<think>\n", unless enable_thinking is false. Then it appends "<think>\n\n</think>\n\n",
# which matches the training target.
CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}

_NO_IMAGE = object()


def user_content(instruction: str = INSTRUCTION, image: Any = _NO_IMAGE) -> list[dict[str, Any]]:
    """Content of the user turn: the instruction, then the image.

    Without ``image``, the image entry is a placeholder (``{"type": "image"}``). The
    processor then gets the pixels through its ``images=`` argument.
    """
    image_part: dict[str, Any] = {"type": "image"}
    if image is not _NO_IMAGE:
        image_part["image"] = image
    return [{"type": "text", "text": instruction}, image_part]


def convert_to_conversation(
    sample: Mapping[str, Any], instruction: str = INSTRUCTION
) -> dict[str, list[dict[str, Any]]]:
    """Turn one ``{"image", "text"}`` row into a user/assistant conversation (training)."""
    return {
        "messages": [
            {"role": "user", "content": user_content(instruction, sample["image"])},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
    }


def inference_messages(instruction: str = INSTRUCTION) -> list[dict[str, Any]]:
    """Messages for one generation request, without the image pixels (inference)."""
    return [{"role": "user", "content": user_content(instruction)}]
