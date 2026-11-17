from mathscribe import tracking


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
