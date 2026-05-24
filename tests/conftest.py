"""Shared pytest fixtures for zmq_inference tests."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from zmq_inference.core.config import ClientConfig, ServerConfig
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse, Prediction
from zmq_inference.utils.image import encode_image


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def server_config() -> ServerConfig:
    """A ServerConfig pointing to a non-production port."""
    return ServerConfig(host="*", port=15576, model_name="mobilenetv2_100", top_k=3)


@pytest.fixture()
def client_config() -> ClientConfig:
    """A ClientConfig pointing to the test server port."""
    return ClientConfig(host="localhost", port=15576, timeout_ms=5000, max_retries=2)


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_pil_image() -> Image.Image:
    """A 4x4 solid-blue RGB PIL image for fast tests."""
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    arr[:, :, 2] = 200  # blue channel
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture()
def tiny_image_b64(tiny_pil_image: Image.Image) -> str:
    """Base64-encoded JPEG of the tiny PIL image."""
    return encode_image(tiny_pil_image)


@pytest.fixture()
def tiny_image_path(tmp_path: object, tiny_pil_image: Image.Image) -> str:
    """Temporary JPEG file on disk; returns the path as a string."""
    import pathlib

    p = pathlib.Path(str(tmp_path)) / "test_image.jpg"
    tiny_pil_image.save(str(p), format="JPEG")
    return str(p)


# ---------------------------------------------------------------------------
# Protocol fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_request(tiny_image_b64: str) -> InferenceRequest:
    """A sample InferenceRequest with a real base64 image."""
    return InferenceRequest(payload=tiny_image_b64, request_id="test-req-001")


@pytest.fixture()
def sample_prediction() -> Prediction:
    return Prediction(label="tabby cat", confidence=0.92345)


@pytest.fixture()
def sample_response(sample_prediction: Prediction, tiny_image_b64: str) -> InferenceResponse:
    return InferenceResponse(
        request_id="test-req-001",
        predictions=[sample_prediction],
        image=tiny_image_b64,
        elapsed_ms=42.5,
    )


# ---------------------------------------------------------------------------
# Mock ZMQ fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_zmq_context() -> MagicMock:
    """A MagicMock standing in for zmq.Context."""
    ctx = MagicMock()
    socket = MagicMock()
    ctx.socket.return_value = socket
    return ctx


@pytest.fixture()
def mock_zmq_socket(mock_zmq_context: MagicMock) -> MagicMock:
    """The socket produced by mock_zmq_context."""
    return mock_zmq_context.socket.return_value
