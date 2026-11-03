from mathscribe.data import INSTRUCTION, convert_to_conversation, to_conversation_batch


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


def test_to_conversation_batch():
    out = to_conversation_batch({"image": ["i0", "i1"], "text": ["t0", "t1"]}, "Do it.")
    assert list(out) == ["messages"]
    assert len(out["messages"]) == 2
    user, assistant = out["messages"][1]
    assert user["content"] == [{"type": "text", "text": "Do it."}, {"type": "image", "image": "i1"}]
    assert assistant["content"][0]["text"] == "t1"
