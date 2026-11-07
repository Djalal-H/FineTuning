"""The training data can exclude the rows of a frozen manifest (the val set)."""

from pathlib import Path

import pytest

from mathscribe.config import DataConfig, load_config
from mathscribe.data.loaders import select_training_rows
from mathscribe.eval.testsets import Manifest, ManifestError

ROOT = Path(__file__).resolve().parents[1]


def test_no_exclusion_keeps_all_rows(source_ds):
    assert select_training_rows(source_ds, DataConfig()) is source_ds


def test_exclusion_removes_the_manifest_rows(source_ds, manifest):
    m = Manifest.load(manifest)
    cfg = DataConfig(dataset="local/toy", split="test", exclude=[str(manifest)])
    rows = select_training_rows(source_ds, cfg)
    assert len(rows) == len(source_ds) - len(m)
    kept_texts = set(rows["text"])
    assert not kept_texts & {source_ds[i]["text"] for i in m.indices}


def test_exclusion_rejects_a_manifest_of_another_dataset(source_ds, manifest):
    cfg = DataConfig(dataset="unsloth/LaTeX_OCR", split="train", exclude=[str(manifest)])
    with pytest.raises(ManifestError):
        select_training_rows(source_ds, cfg)


def test_smoke_config_pins_the_dataset_revision_and_keeps_the_notebook_rows():
    cfg = load_config(ROOT / "configs/train/sft_smoke.yaml")
    val = Manifest.load(ROOT / "data/testsets/latex_ocr_val.json")
    assert cfg.data.revision == val.revision
    # The smoke run reproduces the notebook, so it trains on the full train split.
    assert cfg.data.exclude == []
