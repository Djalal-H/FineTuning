"""Run state that must survive ``--resume`` on a new Colab session.

The trainer saves the state of each ``ExportableState`` callback in the checkpoint's
``trainer_state.json`` (``stateful_callbacks``). The Hub ``last-checkpoint/`` has this
file too, so the state comes back after a disconnect. The state is:

- ``wandb_run_id``: the W&B run to continue (None when tracking is off).
- ``train_seconds``: the train time of all sessions up to this checkpoint. The time
  after the last checkpoint of a lost session is not in it.

``sft.run`` reads the state of the resume checkpoint with ``read_run_state`` before
the trainer starts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

# The class name is the key in trainer_state.json["stateful_callbacks"].
RUN_STATE_CALLBACK = "MathScribeRunState"


def read_run_state(checkpoint: Path | None) -> dict[str, Any]:
    """The run state stored in a checkpoint. Defaults for a new run or an old checkpoint."""
    state: dict[str, Any] = {"wandb_run_id": None, "train_seconds": 0.0}
    if checkpoint is None:
        return state
    state_file = Path(checkpoint) / "trainer_state.json"
    if not state_file.is_file():
        return state
    stored = json.loads(state_file.read_text()).get("stateful_callbacks", {})
    stored = stored.get(RUN_STATE_CALLBACK)
    if isinstance(stored, list):
        stored = stored[-1] if stored else None
    if isinstance(stored, dict):
        args = stored.get("args", {})
        state["wandb_run_id"] = args.get("wandb_run_id")
        state["train_seconds"] = float(args.get("prior_seconds") or 0.0)
    return state


def make_run_state_callback(wandb_run_id: str | None, prior_seconds: float = 0.0) -> Any:
    """A trainer callback that saves the run state in each checkpoint."""
    from transformers import TrainerCallback
    from transformers.trainer_callback import ExportableState

    class MathScribeRunState(TrainerCallback, ExportableState):
        def __init__(self, wandb_run_id: str | None = None, prior_seconds: float = 0.0):
            self.wandb_run_id = wandb_run_id
            self.prior_seconds = float(prior_seconds)
            self.session_start = time.time()

        def on_train_begin(self, args, state, control, **kwargs):
            self.session_start = time.time()

        def total_seconds(self) -> float:
            return self.prior_seconds + (time.time() - self.session_start)

        def state(self) -> dict[str, Any]:
            # Called when a checkpoint is saved: the total up to this checkpoint.
            return {
                "args": {
                    "wandb_run_id": self.wandb_run_id,
                    "prior_seconds": round(self.total_seconds(), 1),
                },
                "attributes": {},
            }

    assert MathScribeRunState.__name__ == RUN_STATE_CALLBACK
    return MathScribeRunState(wandb_run_id, prior_seconds)
