"""Predictor tests that need no GPU, no model download and no API call."""

from types import SimpleNamespace

import pytest

from mathscribe.eval.config import eval_config_from_dict
from mathscribe.eval.predictors import PREDICTORS, AnthropicPredictor, build_predictor

Image = pytest.importorskip("PIL.Image")


class FakeMessages:
    def __init__(self, text="x^2", stop_reason="end_turn", fallback=False):
        self.calls = []
        self.text, self.stop_reason, self.fallback = text, stop_reason, fallback

    def create(self, **kwargs):
        self.calls.append(kwargs)
        iterations = [SimpleNamespace(type="fallback_message")] if self.fallback else []
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking"),
                SimpleNamespace(type="text", text=self.text),
            ],
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, iterations=iterations),
        )


def _api_predictor(monkeypatch, fallbacks=True, **fake):
    cfg = eval_config_from_dict(
        {
            "run_name": "api",
            "model": {
                "backend": "anthropic",
                "name": "claude-opus-5-5",
                "options": {"fallbacks": fallbacks},
            },
            "generation": {"system_prompt": "Only LaTeX.", "batch_size": 2},
        }
    )
    anthropic = pytest.importorskip("anthropic")
    messages = FakeMessages(**fake)
    client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: client)
    return AnthropicPredictor(cfg.model, cfg.generation), messages


def test_backends_are_registered():
    assert set(PREDICTORS) == {"unsloth", "anthropic", "pix2tex", "got_ocr"}


def test_api_request_uses_the_training_order_and_the_fallback_beta(monkeypatch):
    predictor, messages = _api_predictor(monkeypatch)
    rows = [{"image": Image.new("RGB", (20, 10)), "source_index": i} for i in range(3)]
    assert predictor.predict(rows) == ["x^2"] * 3

    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "Only LaTeX."
    assert call["output_config"] == {"effort": "low"}
    assert "temperature" not in call
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    text, image = call["messages"][0]["content"]
    assert text["type"] == "text" and image["type"] == "image"
    assert image["source"]["media_type"] == "image/png"

    info = predictor.runtime_info()
    assert info["requests"] == 3
    assert info["input_tokens"] == 300
    assert info["fallback_rows"] == 0


def test_api_counts_fallbacks_and_stops(monkeypatch):
    predictor, _ = _api_predictor(monkeypatch, fallback=True, stop_reason="max_tokens")
    predictor.predict([{"image": Image.new("RGB", (5, 5)), "source_index": 0}])
    info = predictor.runtime_info()
    assert info["fallback_rows"] == 1
    assert info["max_tokens_stops"] == 1


def test_api_without_fallbacks_uses_the_plain_endpoint(monkeypatch):
    predictor, messages = _api_predictor(monkeypatch, fallbacks=False)
    predictor.predict([{"image": Image.new("RGB", (5, 5)), "source_index": 0}])
    assert "betas" not in messages.calls[0]
    assert "fallbacks" not in messages.calls[0]


def test_unknown_backend_is_an_error():
    cfg = eval_config_from_dict({"run_name": "r"})
    cfg.model.backend = "nope"
    with pytest.raises(ValueError, match="Unknown backend"):
        build_predictor(cfg.model, cfg.generation)


def test_pix2tex_failure_on_one_image_gives_an_empty_prediction():
    from mathscribe.eval.predictors import Pix2TexPredictor

    predictor = Pix2TexPredictor.__new__(Pix2TexPredictor)  # no model download

    def model(image):
        if image.size == (1, 1):
            raise RuntimeError("empty image")
        return "x"

    predictor.model = model
    predictor.failed_rows = []
    predictor.torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    rows = [
        {"image": Image.new("RGB", (5, 5)), "source_index": 0},
        {"image": Image.new("RGB", (1, 1)), "source_index": 7},
    ]
    assert predictor.predict(rows) == ["x", ""]
    assert predictor.runtime_info() == {"gpu": "CPU", "failed_rows": [7]}
