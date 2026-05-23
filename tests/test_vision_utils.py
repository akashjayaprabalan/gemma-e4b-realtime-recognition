from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from app.vision_utils import ImageInputError, parse_model_output, prepare_frame


def test_parse_plain_json() -> None:
    payload = parse_model_output(
        '{"caption":"A mug on a desk.","labels":["mug","desk"],'
        '"objects":["mug"],"text_seen":[],"alerts":[]}'
    )

    assert payload["caption"] == "A mug on a desk."
    assert payload["labels"] == ["mug", "desk"]
    assert payload["objects"] == ["mug"]


def test_parse_json_code_fence() -> None:
    payload = parse_model_output(
        '```json\n{"caption":"A screen.","labels":"monitor","alerts":"glare"}\n```'
    )

    assert payload["caption"] == "A screen."
    assert payload["labels"] == ["monitor"]
    assert payload["alerts"] == ["glare"]


def test_parse_fallback_uses_caption() -> None:
    payload = parse_model_output("A person is holding a phone.")

    assert payload["caption"] == "A person is holding a phone."
    assert payload["labels"] == []


def test_parse_embedded_json_ignores_trailing_text() -> None:
    payload = parse_model_output(
        'Result: {"caption":"A box.","labels":["box"]} trailing note'
    )

    assert payload["caption"] == "A box."
    assert payload["labels"] == ["box"]


def test_prepare_frame_resizes_and_writes_jpeg() -> None:
    source = BytesIO()
    Image.new("RGB", (1600, 1200), "green").save(source, format="PNG")

    path, info = prepare_frame(source.getvalue())

    try:
        assert path.exists()
        assert info["original_width"] == 1600
        assert info["original_height"] == 1200
        assert max(info["width"], info["height"]) <= 896
    finally:
        path.unlink(missing_ok=True)


def test_prepare_frame_rejects_bad_bytes() -> None:
    with pytest.raises(ImageInputError):
        prepare_frame(b"not an image")
