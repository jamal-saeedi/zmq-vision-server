"""Tests for zmq_inference.utils.image encode/decode helpers."""

from __future__ import annotations

import base64
import io
import pathlib

import numpy as np
import pytest
from PIL import Image

from zmq_inference.utils.image import decode_image, encode_image, preprocess_for_display


class TestEncodeImage:
    def test_encode_pil_image_returns_string(self, tiny_pil_image: Image.Image) -> None:
        result = encode_image(tiny_pil_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encode_pil_image_is_valid_base64(self, tiny_pil_image: Image.Image) -> None:
        result = encode_image(tiny_pil_image)
        # Should not raise
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_from_path(self, tiny_image_path: str) -> None:
        result = encode_image(tiny_image_path)
        assert isinstance(result, str)
        # Verify it is a readable image.
        img = Image.open(io.BytesIO(base64.b64decode(result)))
        assert img is not None

    def test_encode_from_pathlib_path(self, tiny_image_path: str) -> None:
        result = encode_image(pathlib.Path(tiny_image_path))
        assert isinstance(result, str)

    def test_encode_rgba_converts_to_rgb(self) -> None:
        rgba = Image.new("RGBA", (8, 8), color=(100, 150, 200, 128))
        result = encode_image(rgba)
        img = decode_image(result)
        assert img.mode == "RGB"

    def test_encode_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            encode_image("/nonexistent/path/image.jpg")

    def test_encode_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError):
            encode_image(12345)  # type: ignore[arg-type]


class TestDecodeImage:
    def test_decode_returns_pil_image(self, tiny_image_b64: str) -> None:
        result = decode_image(tiny_image_b64)
        assert isinstance(result, Image.Image)

    def test_decode_mode_is_rgb(self, tiny_image_b64: str) -> None:
        result = decode_image(tiny_image_b64)
        assert result.mode == "RGB"

    def test_encode_decode_roundtrip(self, tiny_pil_image: Image.Image) -> None:
        b64 = encode_image(tiny_pil_image)
        restored = decode_image(b64)
        # Dimensions should be preserved (4x4).
        assert restored.size == tiny_pil_image.size

    def test_roundtrip_pixel_values_close(self, tiny_pil_image: Image.Image) -> None:
        b64 = encode_image(tiny_pil_image)
        restored = decode_image(b64)
        orig_arr = np.array(tiny_pil_image.convert("RGB"))
        rest_arr = np.array(restored)
        # JPEG is lossy — allow a small tolerance.
        assert np.abs(orig_arr.astype(int) - rest_arr.astype(int)).max() < 30


class TestPreprocessForDisplay:
    def test_resize(self, tiny_pil_image: Image.Image) -> None:
        result = preprocess_for_display(tiny_pil_image, size=(32, 32))
        assert result.size == (32, 32)

    def test_returns_rgb(self) -> None:
        gray = Image.new("L", (10, 10), color=128)
        result = preprocess_for_display(gray)
        assert result.mode == "RGB"

    def test_default_size_is_224(self, tiny_pil_image: Image.Image) -> None:
        result = preprocess_for_display(tiny_pil_image)
        assert result.size == (224, 224)
