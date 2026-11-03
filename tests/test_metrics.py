import math

import pytest

from mathscribe.eval.metrics import (
    aggregate,
    aggregate_by_tag,
    corpus_bleu,
    difficulty_tags,
    levenshtein,
    score_sample,
)


@pytest.mark.parametrize(
    ("a", "b", "d"),
    [("", "", 0), ("abc", "", 3), ("", "ab", 2), ("kitten", "sitting", 3), ("abc", "abc", 0)],
)
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d
    assert levenshtein(b, a) == d


def test_levenshtein_on_token_lists():
    assert levenshtein(["\\frac", "a", "b"], ["\\frac", "a", "c"]) == 1


def test_bleu_identical_is_one():
    tokens = ["a", "b", "c", "d", "e"]
    assert corpus_bleu([tokens], [tokens]) == pytest.approx(1.0)


def test_bleu_disjoint_is_zero():
    assert corpus_bleu([["a", "b", "c", "d"]], [["w", "x", "y", "z"]]) == 0.0


def test_bleu_hand_computed():
    # pred has 6 tokens, ref has 6. Clipped n-gram matches: 5/6, 3/5, 1/4, 0/3 -> 0
    # with no smoothing. Use max_n=3 for a non-zero hand value.
    pred = ["a", "b", "c", "x", "d", "e"]
    ref = ["a", "b", "c", "y", "d", "e"]
    p1, p2, p3 = 5 / 6, 3 / 5, 1 / 4
    expected = math.exp((math.log(p1) + math.log(p2) + math.log(p3)) / 3)
    assert corpus_bleu([pred], [ref], max_n=3) == pytest.approx(expected)
    assert corpus_bleu([pred], [ref]) == 0.0


def test_bleu_brevity_penalty():
    ref = ["a", "b", "c", "d", "e", "f", "g", "h"]
    pred = ref[:4]
    # All n-grams of pred match, so BLEU = BP = exp(1 - 8/4).
    assert corpus_bleu([pred], [ref]) == pytest.approx(math.exp(1 - 8 / 4))


def test_bleu_clips_repeated_ngrams():
    # "the the the the" against "the cat": unigram precision 1/4, not 4/4.
    assert corpus_bleu([["the"] * 4], [["the", "cat"]], max_n=1) == pytest.approx(0.25)


def test_score_sample_exact_ignores_only_whitespace():
    s = score_sample(r"\frac{a}{b}", r"\frac { a } { b }")
    assert s["exact_match"] and s["normalized_match"]
    assert s["cer"] == 0.0

    s = score_sample(r"\left( x \right)", "( x )")
    assert not s["exact_match"]
    assert s["normalized_match"]


def test_score_sample_cer_uses_the_canonical_form():
    s = score_sample("x ^ 3", "x ^ { 2 }")  # canonical: "x ^ 3" vs "x ^ 2"
    assert s["char_edits"] == 1
    assert s["ref_chars"] == len("x ^ 2")
    assert s["cer"] == pytest.approx(1 / 5)


def test_score_sample_empty_prediction():
    s = score_sample("", "x")
    assert s["empty"] and not s["normalized_match"]
    assert s["cer"] == 1.0


def test_aggregate_is_corpus_level():
    scores = [score_sample("a", "a"), score_sample("b c", "b d e f")]
    agg = aggregate(scores)
    assert agg["n"] == 2
    assert agg["exact_match"] == 0.5
    assert agg["normalized_match"] == 0.5
    # edits: 0 + levenshtein("b c", "b d e f") = 5; ref chars: 1 + 7
    assert agg["cer"] == pytest.approx(5 / 8)
    assert agg["empty_rate"] == 0.0


def test_aggregate_empty():
    assert aggregate([]) == {"n": 0}


def test_difficulty_tags():
    assert difficulty_tags("x") == ["len:short"]
    assert difficulty_tags(" ".join(["a"] * 40)) == ["len:medium"]
    assert difficulty_tags(" ".join(["a"] * 60)) == ["len:long"]
    assert difficulty_tags(r"\begin{array}{c} a \\ b \end{array}") == ["len:short", "matrix"]
    assert difficulty_tags(r"\frac { \frac a b } c") == ["len:short", "nested_frac"]


def test_aggregate_by_tag_counts_a_sample_in_each_of_its_tags():
    scores = [score_sample("a", "a"), score_sample("b", "c")]
    out = aggregate_by_tag(scores, [["len:short", "matrix"], ["len:short"]])
    assert out["len:short"]["n"] == 2
    assert out["matrix"]["n"] == 1
    assert out["matrix"]["exact_match"] == 1.0
