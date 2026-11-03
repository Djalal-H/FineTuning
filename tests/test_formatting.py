from mathscribe.data import INSTRUCTION, ConversationDataset, convert_to_conversation


def test_instruction_matches_notebook():
    assert INSTRUCTION == "Write the LaTeX representation for this image."


def test_convert_to_conversation_structure():
    image = object()  # stands in for a PIL image
    conv = convert_to_conversation({"image": image, "text": r"\frac { a } { b }"})

    user, assistant = conv["messages"]
    assert user["role"] == "user"
    assert user["content"] == [
        {"type": "text", "text": INSTRUCTION},
        {"type": "image", "image": image},
    ]
    assert assistant == {
        "role": "assistant",
        "content": [{"type": "text", "text": r"\frac { a } { b }"}],
    }


def test_convert_to_conversation_custom_instruction():
    conv = convert_to_conversation({"image": None, "text": "x"}, instruction="Transcribe.")
    assert conv["messages"][0]["content"][0]["text"] == "Transcribe."


def test_conversation_dataset_is_lazy_and_ordered():
    reads = []

    class Rows:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            reads.append(idx)
            return {"image": idx, "text": f"t{idx}"}

    ds = ConversationDataset(Rows())
    assert len(ds) == 3
    assert reads == []  # nothing converted up front

    item = ds[2]
    assert reads == [2]
    assert item["messages"][1]["content"][0]["text"] == "t2"
    assert item["messages"][0]["content"][1]["image"] == 2
