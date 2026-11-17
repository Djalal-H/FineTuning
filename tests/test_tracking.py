import sys
import types

import pytest

from mathscribe import tracking


def _fake_wandb(monkeypatch, login):
    """Put a fake wandb module in place. ``login`` is the fake wandb.login."""
    mod = types.ModuleType("wandb")
    mod.login = login
    mod.errors = types.SimpleNamespace(UsageError=type("UsageError", (Exception,), {}))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return mod


def test_ensure_login_new_wandb(monkeypatch):
    calls = []

    def login(prompt=True, verify=True):
        calls.append((prompt, verify))
        return True

    _fake_wandb(monkeypatch, login)
    tracking.ensure_login()
    assert calls == [(False, False)]


def test_ensure_login_old_wandb_with_key(monkeypatch):
    calls = []

    def login(verify=True):  # no `prompt` argument, like the wandb on Kaggle
        calls.append(verify)
        return True

    _fake_wandb(monkeypatch, login)
    monkeypatch.setenv("WANDB_API_KEY", "key")
    tracking.ensure_login()
    assert calls == [False]


def test_ensure_login_old_wandb_without_key_stops(monkeypatch):
    def login(verify=True):
        raise AssertionError("login() must not run without a key")

    _fake_wandb(monkeypatch, login)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="no login"):
        tracking.ensure_login()


def test_flatten():
    assert tracking.flatten("x", {"a": 1, "b": {"c": 2, "d": [1, 2]}}) == {
        "x/a": 1,
        "x/b/c": 2,
        "x/b/d": [1, 2],
    }


def test_eval_summary():
    result = {
        "metrics": {"n": 4, "normalized_match": 0.5},
        "ci95": {"normalized_match": [0.25, 0.75]},
        "buckets": {"len:short": {"n": 3, "normalized_match": 0.66}},
        "runtime": {"gpu": "Tesla T4", "cost": {"gpu_hours": 0.1, "usd": 0.035}},
        "testset": {"name": "toy", "n": 4, "sha256": "abc"},
    }
    s = tracking.eval_summary(result)
    assert s["eval/normalized_match"] == 0.5
    assert s["eval_ci95/normalized_match_low"] == 0.25
    assert s["eval_bucket/len:short/n"] == 3
    assert s["runtime/gpu"] == "Tesla T4"
    assert "runtime/cost" not in s
    assert s["cost/usd"] == 0.035
    assert s["testset/name"] == "toy"
