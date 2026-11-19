import json

import pytest

from mathscribe.train.run_state import RUN_STATE_CALLBACK, make_run_state_callback, read_run_state


def _ckpt(tmp_path, stateful):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(
        json.dumps({"global_step": 100, "stateful_callbacks": stateful})
    )
    return ckpt


NEW_RUN = {"wandb_run_id": None, "train_seconds": 0.0}


def test_new_run_and_missing_state(tmp_path):
    assert read_run_state(None) == NEW_RUN
    assert read_run_state(tmp_path / "missing") == NEW_RUN


def test_read_run_state(tmp_path):
    args = {"wandb_run_id": "abc123", "prior_seconds": 3600.5}
    ckpt = _ckpt(tmp_path, {RUN_STATE_CALLBACK: {"args": args, "attributes": {}}})
    assert read_run_state(ckpt) == {"wandb_run_id": "abc123", "train_seconds": 3600.5}


def test_list_form_and_old_checkpoints(tmp_path):
    (tmp_path / "a").mkdir()
    ckpt = _ckpt(tmp_path / "a", {RUN_STATE_CALLBACK: [{"args": {"wandb_run_id": "x"}}]})
    assert read_run_state(ckpt) == {"wandb_run_id": "x", "train_seconds": 0.0}
    (tmp_path / "b").mkdir()
    # A checkpoint from before the callback existed (the Phase 0 smoke run).
    old = _ckpt(tmp_path / "b", {"TrainerControl": {"args": {}}})
    assert read_run_state(old) == NEW_RUN


def test_callback_state_adds_the_session_time(monkeypatch):
    pytest.importorskip("transformers")
    from mathscribe.train import run_state

    clock = iter([1000.0, 1000.0, 1600.0])
    monkeypatch.setattr(run_state.time, "time", lambda: next(clock))
    cb = make_run_state_callback("abc123", prior_seconds=3600.0)  # clock: 1000 (init)
    assert type(cb).__name__ == RUN_STATE_CALLBACK
    cb.on_train_begin(None, None, None)  # clock: 1000
    state = cb.state()  # clock: 1600, so 600 s in this session
    assert state == {"args": {"wandb_run_id": "abc123", "prior_seconds": 4200.0}, "attributes": {}}
    # The trainer restores a stateful callback with type(cb)(**args).
    monkeypatch.setattr(run_state.time, "time", lambda: 0.0)
    again = type(cb)(**state["args"])
    assert (again.wandb_run_id, again.prior_seconds) == ("abc123", 4200.0)


def _guard_input(logs):
    from types import SimpleNamespace

    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=7)
    return state, control, logs


@pytest.mark.parametrize("key", ["loss", "eval_loss", "grad_norm"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_guard_stops_on_a_non_finite_value(key, bad):
    from mathscribe.train.run_state import make_nan_guard_callback

    guard = make_nan_guard_callback()
    state, control, logs = _guard_input({key: bad, "learning_rate": 0.0})
    guard.on_log(None, state, control, logs=logs)
    assert control.should_training_stop is True
    assert (guard.bad_step, guard.bad_metric) == (7, key)


def test_nan_guard_ignores_normal_logs():
    from mathscribe.train.run_state import make_nan_guard_callback

    guard = make_nan_guard_callback()
    for logs in ({"loss": 0.5, "grad_norm": 1.2}, {"learning_rate": 1e-4}, None):
        state, control, logs = _guard_input(logs)
        guard.on_log(None, state, control, logs=logs)
        assert control.should_training_stop is False
    assert guard.bad_step is None
