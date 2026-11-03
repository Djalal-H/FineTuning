import json

from mathscribe.eval.leaderboard import render_leaderboard, write_leaderboard


def _result(label, nm, n=100, sha="a" * 64, name="latex_ocr_test"):
    return {
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
