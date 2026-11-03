"""Shared fixtures: a small local dataset and its frozen manifest (no Hub access)."""

import pytest

TEXTS = [
    r"\frac { a } { b }",
    "x ^ { 2 }",
    r"\left( y \right)",
    r"\begin{array}{c} a \\ b \end{array}",
    r"\frac { \frac { 1 } { 2 } } { c }",
    "a + b = c",
]


@pytest.fixture
def texts():
    return list(TEXTS)


@pytest.fixture
def source_ds():
    datasets = pytest.importorskip("datasets")
    image_mod = pytest.importorskip("PIL.Image")
    images = [image_mod.new("RGB", (30 + 5 * i, 20), color=(i, 0, 0)) for i in range(len(TEXTS))]
    return datasets.Dataset.from_dict(
        {"image": images, "text": TEXTS},
        features=datasets.Features({"image": datasets.Image(), "text": datasets.Value("string")}),
    )


@pytest.fixture
def manifest(source_ds, tmp_path):
    from mathscribe.eval.testsets import Manifest, hash_rows, sample_indices, subset_hash

    indices = sample_indices(len(source_ds), 4, seed=7)
    hashes = hash_rows(source_ds.select(indices))
    m = Manifest(
        name="toy",
        dataset="local/toy",
        revision="0" * 40,
        split="test",
        seed=7,
        source_rows=len(source_ds),
        indices=indices,
        row_hashes=hashes,
        sha256=subset_hash(indices, hashes),
    )
    path = tmp_path / "toy.json"
    path.write_text(m.to_json())
    return path
