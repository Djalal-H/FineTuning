import math

import pytest

from mathscribe.eval.bootstrap import (
    ci,
    paired_diff,
    per_row_scores,
    rate,
    ratio,
    run_ci,
    run_statistics,
)
from mathscribe.eval.metrics import aggregate, score_sample


def test_ci_is_seeded_and_contains_the_estimate():
    values = [1] * 57 + [0] * 43
    a = ci(rate(values), len(values))
    assert a == ci(rate(values), len(values))
    assert a[0] < 0.57 < a[1]


def test_ci_of_a_constant_column_has_no_width():
    assert ci(rate([1] * 50), 50) == (1.0, 1.0)


def test_ci_width_is_close_to_the_normal_approximation():
    # p = 0.5, n = 1000: 1.96 * sqrt(p (1 - p) / n) = 3.1 points on each side.
    values = [1, 0] * 500
    lo, hi = ci(rate(values), len(values))
    half_width = (hi - lo) / 2
    assert half_width == pytest.approx(1.96 * math.sqrt(0.25 / 1000), abs=0.005)


def test_ci_is_narrower_with_more_rows():
    small = ci(rate([1, 0] * 50), 100)
    large = ci(rate([1, 0] * 500), 1000)
    assert large[1] - large[0] < small[1] - small[0]


def test_ratio_is_a_corpus_ratio():
    stat = ratio([1, 3], [10, 10])
    assert stat([0, 1]) == pytest.approx(4 / 20)
    assert stat([1, 1]) == pytest.approx(6 / 20)


def test_paired_diff_of_identical_runs_is_zero():
    values = [1, 0, 1, 1, 0] * 20
    assert paired_diff(rate(values), rate(values), len(values)) == (0.0, 0.0, 0.0)


def test_paired_diff_detects_a_consistent_gain():
    # B is correct on 50 rows. A is correct on the same 50 rows plus 10 more.
    b = [1] * 50 + [0] * 50
    a = [1] * 60 + [0] * 40
    delta, lo, hi = paired_diff(rate(a), rate(b), 100)
    assert delta == pytest.approx(0.10)
    assert 0 < lo < delta < hi


def test_pairing_makes_the_interval_narrower_than_independent_runs():
    # Same 10-point gain as above. Paired, the rows that both runs get right cancel out.
    b = [1] * 50 + [0] * 50
    a = [1] * 60 + [0] * 40
    _, lo, hi = paired_diff(rate(a), rate(b), 100)
    lo_a, hi_a = ci(rate(a), 100)
    lo_b, hi_b = ci(rate(b), 100)
    assert hi - lo < (hi_a - lo_a) + (hi_b - lo_b)


@pytest.mark.parametrize(
    "fn", [lambda: ci(rate([]), 0), lambda: paired_diff(rate([]), rate([]), 0)]
)
def test_empty_input_is_an_error(fn):
    with pytest.raises(ValueError):
        fn()


def test_per_row_scores_round_trip_to_the_aggregate_metrics():
    scores = [
        score_sample("a", "a"),
        score_sample("b c", "b d e f"),
        score_sample(r"\left( x \right)", "( x )"),
    ]
    per_row = per_row_scores(scores)
    assert per_row["normalized_match"] == "101"
    assert per_row["exact_match"] == "100"

    n, stats = run_statistics(per_row)
    agg = aggregate(scores)
    everything = list(range(n))
    for name in ("exact_match", "normalized_match", "cer"):
        assert stats[name](everything) == pytest.approx(agg[name])

    intervals = run_ci(per_row)
    assert set(intervals) == {"exact_match", "normalized_match", "cer"}
    assert all(lo <= hi for lo, hi in intervals.values())


def _render_score(text_ok, compiles, match):
    s = score_sample("a" if text_ok else "b", "a")
    s.update(
        compiles=compiles,
        render_match=match,
        render_similarity=None if match is None else float(match),
    )
    return s


def test_render_columns_skip_rows_whose_reference_does_not_render():
    scores = [
        _render_score(True, True, True),
        _render_score(False, True, False),
        _render_score(False, False, None),  # the reference does not render
    ]
    per_row = per_row_scores(scores)
    assert per_row["compiles"] == "110"
    assert per_row["render_match"] == "10-"
    assert per_row["render_similarity"] == [1.0, 0.0, None]

    n, stats = run_statistics(per_row)
    everything = list(range(n))
    agg = aggregate(scores)
    assert stats["render_match"](everything) == pytest.approx(agg["render_match"]) == 0.5
    assert stats["compile_rate"](everything) == pytest.approx(agg["compile_rate"])
    assert set(run_ci(per_row)) == {
        "exact_match",
        "normalized_match",
        "cer",
        "compile_rate",
        "render_match",
    }


def test_text_only_runs_have_no_render_columns():
    per_row = per_row_scores([score_sample("a", "a")])
    assert "render_match" not in per_row
    assert "render_match" not in run_statistics(per_row)[1]
