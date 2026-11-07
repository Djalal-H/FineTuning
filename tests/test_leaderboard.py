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


def test_render_columns():
    plain = _with_rows(_result("Plain", 0, run_name="zs"), "1" * 50 + "0" * 50)
    board = render_leaderboard([plain], baseline_run="zs")
    row = next(line for line in board.splitlines() if line.startswith("| Plain |"))
    cells = [c.strip() for c in row.split("|")]
    header = [c.strip() for c in board.splitlines()[board.splitlines().index(row) - 2].split("|")]
    assert cells[header.index("Compile")] == "-"
    assert cells[header.index("Render match")] == "**-**"

    def with_render(result, column, renderer="r1"):
        result["per_row"]["compiles"] = "1" * len(column)
        result["per_row"]["render_match"] = column
        result["metrics"]["compile_rate"] = 1.0
        valid = [c for c in column if c != "-"]
        result["metrics"]["render_match"] = valid.count("1") / len(valid)
        result["renderer"] = renderer
        return result

    base = with_render(
        _with_rows(_result("Base", 0, run_name="zs"), "1" * 100), "1" * 40 + "0" * 59 + "-"
    )
    new = with_render(
        _with_rows(_result("New", 0, run_name="sft"), "1" * 100), "1" * 70 + "0" * 29 + "-"
    )
    other = with_render(
        _with_rows(_result("Other", 0, run_name="o"), "1" * 100), "1" * 99 + "-", "r2"
    )
    board = render_leaderboard([base, new, other], baseline_run="zs")
    lines = board.splitlines()
    header = [
        c.strip()
        for c in next(line for line in lines if line.startswith("| Model | Exact")).split("|")
    ]
    new_row = [
        c.strip() for c in next(line for line in lines if line.startswith("| New |")).split("|")
    ]
    assert new_row[header.index("Compile")] == "100.0"
    assert new_row[header.index("Render match")].startswith("**70.7")
    assert new_row[header.index("Δ render")].startswith("+30.3 [")
    assert new_row[header.index("Δ render")].endswith("*")
    # A different renderer version: no paired render delta.
    other_row = [
        c.strip() for c in next(line for line in lines if line.startswith("| Other |")).split("|")
    ]
    assert other_row[header.index("Δ render")] == "-"


def test_uncommitted_runs_are_marked():
    clean = _result("Clean", 0.5)
    dirty = _result("Dirty", 0.4)
    dirty["git_dirty"] = True
    board = render_leaderboard([clean, dirty])
    assert "| `0123456` | 2026-11-03 |" in _row(board, "| Clean |")
    assert "| `0123456`+ | 2026-11-03 |" in _row(board, "| Dirty |")


def _row(board, prefix):
    return next(line for line in board.splitlines() if line.startswith(prefix))


def test_runs_with_different_scorer_versions_are_not_compared():
    old = _with_rows(_result("Old", 0, run_name="zs"), "1" * 50 + "0" * 50)  # no version: 1
    new = _with_rows(_result("New", 0, run_name="sft"), "1" * 60 + "0" * 40)
    new["scorer_version"] = 2
    board = render_leaderboard([old, new], baseline_run="zs")
    assert _row(board, "| New |").split("|")[4].strip() == "-"
    assert "Older scorer (version < 2), not compared with newer runs: Old." in board

    old["scorer_version"] = 2
    board = render_leaderboard([old, new], baseline_run="zs")
    assert _row(board, "| New |").split("|")[4].strip().startswith("+10.0")
    assert "Older scorer" not in board
