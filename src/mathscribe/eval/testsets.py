"""Frozen test sets: a seeded subset of a HF dataset split, pinned to a revision, with hashes.

A manifest (JSON, in ``data/testsets/``) records the dataset, the commit sha of the
dataset repo, the split, the seed, and the selected rows in the sampled order. Each row
has a hash of its image bytes and its text. The rows are in random order, so the first
N rows (``--subset N``) are also a uniform random sample, and smaller subsets are
prefixes of larger ones.

Usage:
    python -m mathscribe.eval.testsets freeze --dataset unsloth/LaTeX_OCR --split test \\
        --n 1000 --seed 2026 --out data/testsets/latex_ocr_test.json
    python -m mathscribe.eval.testsets verify data/testsets/latex_ocr_test.json
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset

ROW_HASH_CHARS = 16


class ManifestError(ValueError):
    """The test set does not match its manifest."""


@dataclass
class Manifest:
    name: str
    dataset: str
    revision: str
    split: str
    seed: int
    source_rows: int
    indices: list[int]
    row_hashes: list[str]
    sha256: str
    created: str = ""

    def __len__(self) -> int:
        return len(self.indices)

    def prefix(self, n: int | None) -> tuple[list[int], list[str]]:
        """The first ``n`` rows (all rows when ``n`` is None)."""
        if n is None:
            n = len(self)
        if not 0 < n <= len(self):
            raise ManifestError(f"subset must be in 1..{len(self)}, got {n}")
        return self.indices[:n], self.row_hashes[:n]

    def to_json(self) -> str:
        """JSON with one row per line, so that diffs stay readable."""
        head = {
            k: v for k, v in dataclasses.asdict(self).items() if k not in ("indices", "row_hashes")
        }
        lines = ",\n".join(
            f"    [{i}, {json.dumps(h)}]"
            for i, h in zip(self.indices, self.row_hashes, strict=True)
        )
        body = json.dumps(head, indent=2)[:-2]  # drop the closing "\n}"
        return f'{body},\n  "rows": [\n{lines}\n  ]\n}}\n'

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        raw = json.loads(text)
        rows = raw.pop("rows")
        manifest = cls(**raw, indices=[r[0] for r in rows], row_hashes=[r[1] for r in rows])
        if subset_hash(manifest.indices, manifest.row_hashes) != manifest.sha256:
            raise ManifestError(f"manifest {manifest.name!r}: rows do not match its sha256")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.from_json(Path(path).read_text())


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def sample_indices(num_rows: int, n: int, seed: int) -> list[int]:
    """``n`` distinct row indices in a seeded random order."""
    if not 0 < n <= num_rows:
        raise ManifestError(f"n must be in 1..{num_rows}, got {n}")
    return random.Random(seed).sample(range(num_rows), n)


def row_hash(image_bytes: bytes, text: str) -> str:
    h = hashlib.sha256()
    h.update(hashlib.sha256(image_bytes).digest())
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:ROW_HASH_CHARS]


def subset_hash(indices: list[int], row_hashes: list[str]) -> str:
    """sha256 of the ordered rows. It identifies a test set (or a prefix of it)."""
    payload = "\n".join(f"{i}:{h}" for i, h in zip(indices, row_hashes, strict=True))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _image_bytes(image: dict[str, Any]) -> bytes:
    if image.get("bytes") is not None:
        return image["bytes"]
    return Path(image["path"]).read_bytes()


def hash_rows(ds: Dataset) -> list[str]:
    """Row hashes of a dataset with an ``image`` and a ``text`` column."""
    from datasets import Image

    raw = ds.cast_column("image", Image(decode=False))
    return [row_hash(_image_bytes(r["image"]), r["text"]) for r in raw]


# --------------------------------------------------------------------------- #
# Hub access
# --------------------------------------------------------------------------- #
def resolve_revision(dataset: str, revision: str | None = None) -> str:
    """The commit sha of a dataset repo (of ``revision``, or of the main branch)."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(dataset, revision=revision).sha


def _load_split(dataset: str, split: str, revision: str) -> Dataset:
    from datasets import load_dataset

    return load_dataset(dataset, split=split, revision=revision)


def freeze(
    dataset: str, split: str, n: int, seed: int, name: str, revision: str | None = None
) -> Manifest:
    sha = resolve_revision(dataset, revision)
    ds = _load_split(dataset, split, sha)
    indices = sample_indices(len(ds), n, seed)
    hashes = hash_rows(ds.select(indices))
    return Manifest(
        name=name,
        dataset=dataset,
        revision=sha,
        split=split,
        seed=seed,
        source_rows=len(ds),
        indices=indices,
        row_hashes=hashes,
        sha256=subset_hash(indices, hashes),
        created=datetime.date.today().isoformat(),
    )


def load_testset(
    manifest: Manifest, subset: int | None = None, source: Dataset | None = None
) -> Dataset:
    """Load the rows of the manifest (the first ``subset`` rows) and check their hashes.

    ``source`` replaces the Hub download (for tests). The result has the columns
    ``image``, ``text`` and ``source_index``, in the manifest order.
    """
    indices, expected = manifest.prefix(subset)
    if source is None:
        source = _load_split(manifest.dataset, manifest.split, manifest.revision)
    if len(source) != manifest.source_rows:
        raise ManifestError(
            f"{manifest.dataset}/{manifest.split} has {len(source)} rows, "
            f"the manifest expects {manifest.source_rows}"
        )
    rows = source.select(indices)
    actual = hash_rows(rows)
    bad = [i for i, a, e in zip(indices, actual, expected, strict=True) if a != e]
    if bad:
        raise ManifestError(f"{len(bad)} rows do not match the manifest (first: {bad[:5]})")
    return rows.add_column("source_index", indices)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="mathscribe.eval.testsets", description=__doc__.splitlines()[0]
    )
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("freeze", help="Sample a subset and write its manifest")
    f.add_argument("--dataset", default="unsloth/LaTeX_OCR")
    f.add_argument("--split", default="test")
    f.add_argument("--n", type=int, default=1000)
    f.add_argument("--seed", type=int, default=2026)
    f.add_argument("--revision", default=None, help="Dataset revision (default: main)")
    f.add_argument("--name", default=None, help="Test set name (default: the file stem)")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--force", action="store_true", help="Overwrite an existing manifest")

    v = sub.add_parser("verify", help="Download the rows and check them against the manifest")
    v.add_argument("manifest", type=Path)

    args = p.parse_args(argv)
    if args.command == "freeze":
        if args.out.exists() and not args.force:
            raise SystemExit(f"{args.out} exists. A frozen test set must not change (--force).")
        manifest = freeze(
            args.dataset, args.split, args.n, args.seed, args.name or args.out.stem, args.revision
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(manifest.to_json())
        print(f"Wrote {args.out}: {len(manifest)} rows, sha256 {manifest.sha256}")
    else:
        manifest = Manifest.load(args.manifest)
        load_testset(manifest)
        print(f"OK: {manifest.name}, {len(manifest)} rows, sha256 {manifest.sha256}")


if __name__ == "__main__":
    main()
