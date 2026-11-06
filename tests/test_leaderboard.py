import json

from mathscribe.eval.leaderboard import render_leaderboard, write_leaderboard


def _result(label, nm, n=100, sha="a" * 64, name="latex_ocr_test", run_name=None):
    return {
        "run_name": run_name or label,
        "label": label,
        "date": "2026-11-03",
        "git_commit": "0123456789",
        "testset": {
            "name": name,
            "n": n,
            "sha256": sha,
            "dataset": "unsloth/LaTeX_OCR",
            "split": "test",
            "revision": "4da395c8a0253f4f",
        },
        "metrics": {
            "n": n,
            "exact_match": nm / 2,
            "normalized_match": nm,
            "cer": 0.1,
            "bleu": 0.5,
            "empty_rate": 0.0,
        },
        "buckets": {"len:short": {"n": 10, "normalized_match": nm}},
        "runtime": {"gpu": "Tesla T4", "images_per_s": 1.5},
    }


def test_rows_are_ranked_by_normalized_match():
    board = render_leaderboard([_result("B", 0.2), _result("A", 0.9)])
    assert board.index("| A |") < board.index("| B |")
    assert "**90.0**" in board
    assert "| 90.0 (10) | - |" in board


def test_different_subsets_get_different_tables():
    board = render_leaderboard([_result("A", 0.9, n=1000, sha="b" * 64), _result("A", 0.8, n=100)])
    assert board.count("## latex_ocr_test") == 2
    # The larger test set comes first.
    assert board.index("n=1000") < board.index("n=100\n")


def test_empty_leaderboard():
    assert "No results yet" in render_leaderboard([])


def test_write_leaderboard_reads_the_results_dir(tmp_path):
    (tmp_path / "res").mkdir()
    (tmp_path / "res" / "a.json").write_text(json.dumps(_result("A", 0.5)))
    out = write_leaderboard(tmp_path / "res", tmp_path / "board.md")
    assert "| A |" in out.read_text()


def _with_rows(result, column):
    """Add per-row scores and intervals, as run_eval writes them."""
    n = len(column)
    result["metrics"]["n"] = n
    result["metrics"]["normalized_match"] = column.count("1") / n
    result["testset"]["n"] = n
    result["per_row"] = {
        "exact_match": column,
        "normalized_match": column,
        "char_edits": [0 if c == "1" else 3 for c in column],
        "ref_chars": [10] * n,
    }
    result["ci95"] = {"normalized_match": [0.4, 0.6], "cer": [0.1, 0.2]}
    return result


def test_intervals_and_paired_delta_vs_baseline():
    base = _with_rows(_result("Base", 0, run_name="zs"), "1" * 50 + "0" * 50)
    better = _with_rows(_result("Better", 0, run_name="sft"), "1" * 60 + "0" * 40)
    same = _with_rows(_result("Same", 0, run_name="same"), "1" * 50 + "0" * 50)
    board = render_leaderboard([base, better, same], baseline_run="zs")

    assert "**50.0 [40.0, 60.0]**" in board
    assert "| 10.0 [10.0, 20.0] |" in board  # CER with its interval
    better_row = next(line for line in board.splitlines() if line.startswith("| Better |"))
    assert "+10.0 [" in better_row and better_row.split("|")[4].strip().endswith("*")
    same_row = next(line for line in board.splitlines() if line.startswith("| Same |"))
    assert "+0.0 [+0.0, +0.0] |" in same_row
    base_row = next(line for line in board.splitlines() if line.startswith("| Base |"))
    assert "| baseline |" in base_row


def test_delta_is_empty_without_per_row_scores_or_baseline():
    old = _result("Old", 0.5, run_name="zs")  # an older result file: no per_row, no ci95
    new = _with_rows(_result("New", 0, run_name="sft"), "1" * 100)
    board = render_leaderboard([old, new], baseline_run="zs")
    new_row = next(line for line in board.splitlines() if line.startswith("| New |"))
    assert new_row.split("|")[4].strip() == "-"
    old_row = next(line for line in board.splitlines() if line.startswith("| Old |"))
    assert "**50.0**" in old_row  # the value without an interval

    board = render_leaderboard([new], baseline_run="missing")
    new_row = next(line for line in board.splitlines() if line.startswith("| New |"))
    assert new_row.split("|")[4].strip() == "-"
