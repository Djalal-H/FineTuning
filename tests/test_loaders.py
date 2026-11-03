"""as_conversations must give what TRL's SFTTrainer and UnslothVisionDataCollator expect."""

import pytest

datasets = pytest.importorskip("datasets")
Image = pytest.importorskip("PIL.Image")

from mathscribe.data import INSTRUCTION, as_conversations  # noqa: E402


@pytest.fixture
def raw():
    images = [Image.new("RGB", (40 + 10 * i, 20), color=(i, i, i)) for i in range(3)]
    return datasets.Dataset.from_dict(
        {"image": images, "text": ["a", "b", "c"]},
        features=datasets.Features({"image": datasets.Image(), "text": datasets.Value("string")}),
    )


def test_result_is_a_datasets_dataset(raw):
    # TRL >= 1.x raises TypeError for anything else (sft_trainer.py: "must be a `Dataset`").
    ds = as_conversations(raw)
    assert isinstance(ds, datasets.Dataset)
    assert len(ds) == len(raw)


def test_single_row_is_a_conversation_with_a_pil_image(raw):
    item = as_conversations(raw)[1]
    assert list(item) == ["messages"]
    user, assistant = item["messages"]
    assert user["content"][0] == {"type": "text", "text": INSTRUCTION}
    image = user["content"][1]["image"]
    assert isinstance(image, Image.Image)
    assert image.size == (50, 20)
    assert assistant["content"][0]["text"] == "b"


def test_iteration_matches_trl_vision_detection(raw):
    # TRL reads next(iter(ds)). The notebook's list items had only "messages", so
    # TRL's dataset-level vision detection stays off, as in the notebook run.
    sample = next(iter(as_conversations(raw)))
    assert "image" not in sample
    assert "images" not in sample


def test_batched_fetch_keeps_order(raw):
    # torch's DataLoader fetches a list of indices through Dataset.__getitems__.
    items = as_conversations(raw).__getitems__([2, 0])
    assert [it["messages"][1]["content"][0]["text"] for it in items] == ["c", "a"]


def test_underlying_columns_are_unchanged(raw):
    ds = as_conversations(raw, instruction="X")
    assert ds.column_names == ["image", "text"]
    assert ds[0]["messages"][0]["content"][0]["text"] == "X"
