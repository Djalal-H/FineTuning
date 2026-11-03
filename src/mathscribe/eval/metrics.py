"""Text metrics for LaTeX OCR: exact match, normalized match, CER, BLEU, difficulty buckets.

Definitions (all rates are in [0, 1]):

- ``exact_match``: the token sequences are equal. Only whitespace is ignored.
- ``normalized_match``: the canonical forms are equal (see ``normalize.normalize``).
- ``cer``: character edit distance between the canonical forms, divided by the length
  of the reference canonical form. The corpus value is sum(edits) / sum(ref length).
- ``bleu``: corpus BLEU-4 (Papineni et al., 2002) on the canonical tokens. Uniform
  weights, brevity penalty, no smoothing.
- ``empty_rate``: the share of predictions that are empty after ``extract_latex``.

The canonical form removes whitespace and spacing differences, so CER and BLEU measure
the content and not the style of the output.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from mathscribe.eval.normalize import (
    UnbalancedBraces,
    frac_depth,
    normalize_tokens,
    parse,
    tokenize,
)

# Length buckets: the number of canonical reference tokens. The limits are the tertiles
# of the unsloth/LaTeX_OCR test split (7,632 rows).
LENGTH_BUCKETS: list[tuple[str, int]] = [("len:short", 32), ("len:medium", 50)]
LENGTH_LONG = "len:long"


def levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    """Edit distance (insert, delete, substitute; cost 1 each) between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: Iterable[Sequence[str]], references: Iterable[Sequence[str]], max_n: int = 4
) -> float:
    """Corpus BLEU with one reference per prediction."""
    matches = [0] * max_n
    totals = [0] * max_n
    pred_len = ref_len = 0
    for pred, ref in zip(predictions, references, strict=True):
        pred_len += len(pred)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            pred_ngrams = _ngrams(pred, n)
            ref_ngrams = _ngrams(ref, n)
            matches[n - 1] += sum(min(c, ref_ngrams[g]) for g, c in pred_ngrams.items())
            totals[n - 1] += max(len(pred) - n + 1, 0)
    if pred_len == 0 or min(matches) == 0:
        return 0.0
    log_precision = sum(math.log(m / t) for m, t in zip(matches, totals, strict=True)) / max_n
    brevity = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    return brevity * math.exp(log_precision)


def difficulty_tags(reference: str) -> list[str]:
    """Buckets of a reference: one length bucket, plus ``matrix`` and ``nested_frac``."""
    tokens = normalize_tokens(reference)
    tags = [next((name for name, limit in LENGTH_BUCKETS if len(tokens) <= limit), LENGTH_LONG)]
    if "\\begin" in tokens or "\\\\" in tokens:
        tags.append("matrix")
    try:
        if frac_depth(parse(tokens)) >= 2:
            tags.append("nested_frac")
    except UnbalancedBraces:
        pass
    return tags


def score_sample(prediction: str, reference: str) -> dict[str, Any]:
    """Per-sample scores. ``pred_tokens`` and ``ref_tokens`` are the canonical tokens."""
    pred_tokens = normalize_tokens(prediction)
    ref_tokens = normalize_tokens(reference)
    pred_norm = " ".join(pred_tokens)
    ref_norm = " ".join(ref_tokens)
    edits = levenshtein(pred_norm, ref_norm)
    return {
        "exact_match": tokenize(prediction) == tokenize(reference),
        "normalized_match": pred_norm == ref_norm,
        "cer": edits / max(len(ref_norm), 1),
        "char_edits": edits,
        "ref_chars": len(ref_norm),
        "empty": not prediction.strip(),
        "pred_tokens": pred_tokens,
        "ref_tokens": ref_tokens,
    }


def aggregate(scores: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Corpus metrics over the per-sample scores of ``score_sample``."""
    n = len(scores)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "exact_match": sum(s["exact_match"] for s in scores) / n,
        "normalized_match": sum(s["normalized_match"] for s in scores) / n,
        "cer": sum(s["char_edits"] for s in scores) / max(sum(s["ref_chars"] for s in scores), 1),
        "bleu": corpus_bleu([s["pred_tokens"] for s in scores], [s["ref_tokens"] for s in scores]),
        "empty_rate": sum(s["empty"] for s in scores) / n,
    }


def aggregate_by_tag(
    scores: Sequence[dict[str, Any]], tags: Sequence[Sequence[str]]
) -> dict[str, dict[str, float | int]]:
    """``aggregate`` for each difficulty tag. A sample can be in more than one tag."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for score, sample_tags in zip(scores, tags, strict=True):
        for tag in sample_tags:
            groups.setdefault(tag, []).append(score)
    return {tag: aggregate(groups[tag]) for tag in sorted(groups)}
