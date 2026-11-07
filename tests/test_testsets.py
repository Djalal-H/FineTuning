from pathlib import Path

import pytest

from mathscribe.eval.testsets import (
    Manifest,
    ManifestError,
    excluded_complement,
    load_testset,
    overlap_count,
    row_hash,
    sample_indices,
    subset_hash,
)

FROZEN = Path(__file__).resolve().parents[1] / "data" / "testsets" / "latex_ocr_test.json"


def test_sample_indices_is_seeded_distinct_and_unsorted():
    a = sample_indices(1000, 50, seed=1)
    assert a == sample_indices(1000, 50, seed=1)
    assert a != sample_indices(1000, 50, seed=2)
    assert len(set(a)) == 50
    assert a != sorted(a)


def test_smaller_samples_are_not_prefixes_but_manifest_prefixes_are():
    # The manifest stores one order. Its prefixes are the nested subsets.
    order = sample_indices(100, 20, seed=3)
    assert order[:5] == sample_indices(100, 20, seed=3)[:5]


@pytest.mark.parametrize("n", [0, 11])
def test_sample_indices_rejects_bad_n(n):
    with pytest.raises(ManifestError):
        sample_indices(10, n, seed=0)


def test_row_hash_depends_on_image_and_text():
    h = row_hash(b"img", "x")
    assert len(h) == 16
    assert h == row_hash(b"img", "x")
    assert h != row_hash(b"img2", "x")
    assert h != row_hash(b"img", "y")


def test_manifest_json_round_trip(manifest):
    m = Manifest.load(manifest)
    assert Manifest.from_json(m.to_json()) == m
    assert m.prefix(2) == (m.indices[:2], m.row_hashes[:2])
    assert m.prefix(None) == (m.indices, m.row_hashes)
    with pytest.raises(ManifestError):
        m.prefix(5)


def test_manifest_with_edited_rows_is_rejected(manifest):
    text = manifest.read_text()
    m = Manifest.load(manifest)
    tampered = text.replace(m.row_hashes[0], "f" * 16)
    with pytest.raises(ManifestError, match="sha256"):
        Manifest.from_json(tampered)


def test_load_testset_returns_rows_in_manifest_order(manifest, source_ds):
    m = Manifest.load(manifest)
    rows = load_testset(m, subset=3, source=source_ds)
    assert rows.column_names == ["image", "text", "source_index"]
    assert rows["source_index"] == m.indices[:3]
    assert rows["text"] == [source_ds[i]["text"] for i in m.indices[:3]]


def test_load_testset_detects_changed_rows(manifest, source_ds):
    m = Manifest.load(manifest)
    changed = source_ds.map(lambda r, i: {"text": r["text"] + " ."}, with_indices=True)
    with pytest.raises(ManifestError, match="do not match"):
        load_testset(m, source=changed)


def test_load_testset_detects_a_different_source_size(manifest, source_ds):
    with pytest.raises(ManifestError, match="rows"):
        load_testset(Manifest.load(manifest), source=source_ds.select(range(5)))


def test_committed_manifest_is_consistent():
    m = Manifest.load(FROZEN)  # also checks the sha256 of the rows
    assert (m.dataset, m.split, m.seed, len(m)) == ("unsloth/LaTeX_OCR", "test", 2026, 1000)
    assert m.source_rows == 7632
    assert len(m.revision) == 40
    assert len(set(m.indices)) == len(m)
    assert m.sha256 == subset_hash(m.indices, m.row_hashes)


def test_manifest_with_other_column_names(tmp_path, source_ds):
    # CROHME stores the formula in "label", not "text".
    from mathscribe.eval.testsets import hash_rows

    crohme = source_ds.rename_column("text", "label")
    indices = sample_indices(len(crohme), 3, seed=1)
    hashes = hash_rows(crohme.select(indices), "image", "label")
    m = Manifest(
        name="toy_crohme",
        dataset="local/crohme",
        revision="0" * 40,
        split="2014",
        seed=1,
        source_rows=len(crohme),
        indices=indices,
        row_hashes=hashes,
        sha256=subset_hash(indices, hashes),
        text_column="label",
    )
    m = Manifest.from_json(m.to_json())
    assert m.text_column == "label"
    rows = load_testset(m, source=crohme)
    assert rows.column_names == ["image", "text", "source_index"]
    assert rows["text"] == [crohme[i]["label"] for i in indices]
    # Same rows, same hashes as with the default column names.
    assert hashes == hash_rows(source_ds.select(indices))


def test_old_manifests_default_to_image_and_text_columns():
    m = Manifest.load(FROZEN)
    assert (m.image_column, m.text_column) == ("image", "text")


def test_excluded_complement(manifest):
    m = Manifest.load(manifest)
    keep = excluded_complement(6, "local/toy", "test", None, [m])
    assert sorted(keep + m.indices) == list(range(6))
    assert not set(keep) & set(m.indices)
    assert keep == sorted(keep)


@pytest.mark.parametrize(
    ("dataset", "split", "num_rows", "revision", "match"),
    [
        ("other/ds", "test", 6, None, "not other/ds"),
        ("local/toy", "train", 6, None, "not local/toy/train"),
        ("local/toy", "test", 7, None, "7 rows"),
        ("local/toy", "test", 6, "1" * 40, "revision"),
    ],
)
def test_excluded_complement_rejects_another_split(
    manifest, dataset, split, num_rows, revision, match
):
    with pytest.raises(ManifestError, match=match):
        excluded_complement(num_rows, dataset, split, revision, [Manifest.load(manifest)])


def test_overlap_count_uses_the_canonical_form():
    assert overlap_count(["x ^ { 2 }", "a", "b"], ["x^2", "c", "a"]) == 2


@pytest.mark.parametrize("path", sorted(FROZEN.parent.glob("*.json")), ids=lambda p: p.name)
def test_all_committed_manifests_are_consistent(path):
    m = Manifest.load(path)
    assert len(set(m.indices)) == len(m)
    assert all(0 <= i < m.source_rows for i in m.indices)
    assert len(m.revision) == 40


def test_val_and_crohme_manifests():
    val = Manifest.load(FROZEN.parent / "latex_ocr_val.json")
    assert (val.dataset, val.split, len(val), val.source_rows) == (
        "unsloth/LaTeX_OCR",
        "train",
        1000,
        68686,
    )
    test = Manifest.load(FROZEN)
    assert val.revision == test.revision
    crohme = Manifest.load(FROZEN.parent / "crohme2014.json")
    assert (crohme.dataset, crohme.split, len(crohme), crohme.text_column) == (
        "Neeze/CROHME-full",
        "2014",
        986,
        "label",
    )
