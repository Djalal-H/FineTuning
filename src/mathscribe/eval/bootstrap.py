"""Bootstrap confidence intervals for eval metrics (percentile method, seeded).

- ``ci``: the interval of one run's metric. Resample the rows with replacement.
- ``paired_diff``: the interval of (run A - run B) on the same rows. Each resample uses
  the same row indices for both runs, so the row difficulty cancels out. If the interval
  does not include 0, the difference is significant at the 5% level.

A metric is a function of a list of row indices, so the same code works for rates
(normalized match) and for corpus ratios (CER = sum(edits) / sum(ref chars)).
The seed is fixed, so the leaderboard is the same each time it is built.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

RESAMPLES = 2000
SEED = 0
LEVEL = 0.95

Statistic = Callable[[Sequence[int]], float]


def rate(values: Sequence[int | bool]) -> Statistic:
    """The mean of a 0/1 column, for a list of row indices."""
    return lambda idx: sum(values[i] for i in idx) / len(idx)


def ratio(numerators: Sequence[float], denominators: Sequence[float]) -> Statistic:
    """sum(numerators) / sum(denominators), for a list of row indices (corpus CER)."""
    return lambda idx: sum(numerators[i] for i in idx) / max(sum(denominators[i] for i in idx), 1)


def _percentiles(stats: list[float], level: float) -> tuple[float, float]:
    stats.sort()
    tail = (1 - level) / 2
    lo = stats[int(tail * len(stats))]
    hi = stats[min(int((1 - tail) * len(stats)), len(stats) - 1)]
    return lo, hi


def ci(
    stat: Statistic, n: int, resamples: int = RESAMPLES, seed: int = SEED, level: float = LEVEL
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``stat`` over ``n`` rows."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = random.Random(seed)
    rows = range(n)
    stats = [stat(rng.choices(rows, k=n)) for _ in range(resamples)]
    return _percentiles(stats, level)


def paired_diff(
    stat_a: Statistic,
    stat_b: Statistic,
    n: int,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    level: float = LEVEL,
) -> tuple[float, float, float]:
    """(A - B on all rows, low, high). A and B must score the same rows in the same order."""
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = random.Random(seed)
    rows = range(n)
    all_rows = list(rows)
    stats = []
    for _ in range(resamples):
        idx = rng.choices(rows, k=n)
        stats.append(stat_a(idx) - stat_b(idx))
    lo, hi = _percentiles(stats, level)
    return stat_a(all_rows) - stat_b(all_rows), lo, hi


# --------------------------------------------------------------------------- #
# Per-row scores of a run (stored in the result JSON)
# --------------------------------------------------------------------------- #
def _flags(values: Sequence[bool | None]) -> str:
    """0/1 column as a string. "-" is a row with no value (its reference does not render)."""
    return "".join("-" if v is None else ("1" if v else "0") for v in values)


def per_row_scores(scores: Sequence[dict]) -> dict[str, str | list]:
    """The per-row scores that the intervals need, in the test set order.

    0/1 columns are strings ("0110..."), so 1000 rows take about 1 KB each.
    The render columns are there only when the run used the renderer.
    """
    out: dict[str, str | list] = {
        "exact_match": _flags([s["exact_match"] for s in scores]),
        "normalized_match": _flags([s["normalized_match"] for s in scores]),
        "char_edits": [int(s["char_edits"]) for s in scores],
        "ref_chars": [int(s["ref_chars"]) for s in scores],
    }
    if scores and "compiles" in scores[0]:
        out["compiles"] = _flags([s["compiles"] for s in scores])
        out["render_match"] = _flags([s["render_match"] for s in scores])
        out["render_similarity"] = [
            None if s["render_similarity"] is None else round(s["render_similarity"], 4)
            for s in scores
        ]
    return out


def run_statistics(per_row: dict) -> tuple[int, dict[str, Statistic]]:
    """(number of rows, a statistic for each metric that the per-row scores support)."""
    em = [c == "1" for c in per_row["exact_match"]]
    nm = [c == "1" for c in per_row["normalized_match"]]
    stats = {
        "exact_match": rate(em),
        "normalized_match": rate(nm),
        "cer": ratio(per_row["char_edits"], per_row["ref_chars"]),
    }
    if "render_match" in per_row:
        stats["compile_rate"] = rate([c == "1" for c in per_row["compiles"]])
        # Rows whose reference does not render ("-") are not in the denominator.
        stats["render_match"] = ratio(
            [c == "1" for c in per_row["render_match"]],
            [c != "-" for c in per_row["render_match"]],
        )
    return len(nm), stats


def run_ci(per_row: dict) -> dict[str, list[float]]:
    """The 95% interval of each metric that the per-row scores support."""
    n, stats = run_statistics(per_row)
    return {name: [round(v, 4) for v in ci(stat, n)] for name, stat in stats.items()}
