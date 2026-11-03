from mathscribe.data import (
    CHAT_TEMPLATE_KWARGS,
    INSTRUCTION,
    convert_to_conversation,
    inference_messages,
    to_conversation_batch,
    user_content,
)


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


def test_inference_prompt_uses_the_training_order():
    train_user = convert_to_conversation({"image": "IMG", "text": "x"})["messages"][0]
    (infer_user,) = inference_messages()
    assert infer_user["role"] == train_user["role"] == "user"
    assert [part["type"] for part in infer_user["content"]] == ["text", "image"]
    assert [part["type"] for part in train_user["content"]] == ["text", "image"]
    assert infer_user["content"][0] == train_user["content"][0]
    # The pixels go through the processor's images= argument, not the message.
    assert "image" not in infer_user["content"][1]


def test_user_content_keeps_an_explicit_none_image():
    assert user_content("Do it.", None)[1] == {"type": "image", "image": None}


def test_chat_template_kwargs_turn_thinking_off():
    assert CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}
