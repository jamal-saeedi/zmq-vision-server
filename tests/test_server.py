"""Tests for zmq_inference.server.base.RequestHandler and BaseInferenceServer."""

from __future__ import annotations

import json
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.protocol import InferenceResponse, Prediction
from zmq_inference.server.base import BaseInferenceServer, RequestHandler


# ---------------------------------------------------------------------------
# Concrete stub server for testing (no real model)
# ---------------------------------------------------------------------------


class StubServer(BaseInferenceServer):
    """Minimal server stub that returns canned predictions without a real model."""

    def __init__(self, config: ServerConfig, canned: List[Prediction]) -> None:
        super().__init__(config)
        self._canned = canned

    def load_model(self) -> None:
        pass  # no-op

    def preprocess(self, img: Image.Image) -> Any:
        return img  # pass-through

    def predict(self, tensor: Any) -> List[Prediction]:
        return self._canned


# ---------------------------------------------------------------------------
# RequestHandler tests
# ---------------------------------------------------------------------------


class TestRequestHandler:
    def _make_handler(
        self,
        server: StubServer,
        raw_msg: dict,
        client_id: bytes = b"client-1",
    ) -> RequestHandler:
        ctx = MagicMock()
        worker_socket = MagicMock()
        ctx.socket.return_value = worker_socket
        handler = RequestHandler(ctx, client_id, raw_msg, server)
        return handler

    def test_process_returns_valid_response(
        self,
        server_config: ServerConfig,
        tiny_image_b64: str,
    ) -> None:
        canned = [Prediction(label="cat", confidence=0.99)]
        server = StubServer(server_config, canned)

        raw_msg = {
            "payload": tiny_image_b64,
            "request_id": "test-123",
            "metadata": {},
        }

        # Directly call the core logic via run() but intercept the socket send.
        ctx = MagicMock()
        worker_socket = MagicMock()
        ctx.socket.return_value = worker_socket

        handler = RequestHandler(ctx, b"client-1", raw_msg, server)
        handler.run()  # Runs in-thread (synchronous since Thread.start not called)

        # Verify the worker sent back a response.
        assert worker_socket.send.call_count >= 1  # identity frame
        assert worker_socket.send_string.call_count == 1

        sent_json: str = worker_socket.send_string.call_args[0][0]
        response = InferenceResponse.from_json(sent_json)
        assert response.request_id == "test-123"
        assert response.ok
        assert response.predictions[0].label == "cat"
        assert response.elapsed_ms >= 0

    def test_process_error_returns_error_response(
        self,
        server_config: ServerConfig,
    ) -> None:
        """A malformed payload should yield an InferenceResponse with error set."""

        class FailingServer(StubServer):
            def preprocess(self, img: Image.Image) -> Any:
                raise ValueError("simulated preprocess failure")

        server = FailingServer(server_config, [])
        ctx = MagicMock()
        worker_socket = MagicMock()
        ctx.socket.return_value = worker_socket

        raw_msg = {
            "payload": "notvalidbase64!!!",
            "request_id": "bad-req",
            "metadata": {},
        }

        handler = RequestHandler(ctx, b"client-bad", raw_msg, server)
        handler.run()

        sent_json: str = worker_socket.send_string.call_args[0][0]
        response = InferenceResponse.from_json(sent_json)
        assert not response.ok
        assert response.error is not None

    def test_process_preserves_request_id(
        self,
        server_config: ServerConfig,
        tiny_image_b64: str,
    ) -> None:
        canned = [Prediction(label="dog", confidence=0.8)]
        server = StubServer(server_config, canned)
        ctx = MagicMock()
        worker_socket = MagicMock()
        ctx.socket.return_value = worker_socket

        raw_msg = {
            "payload": tiny_image_b64,
            "request_id": "preserved-id-xyz",
            "metadata": {},
        }

        handler = RequestHandler(ctx, b"c", raw_msg, server)
        handler.run()

        sent_json: str = worker_socket.send_string.call_args[0][0]
        response = InferenceResponse.from_json(sent_json)
        assert response.request_id == "preserved-id-xyz"


# ---------------------------------------------------------------------------
# BaseInferenceServer lifecycle tests
# ---------------------------------------------------------------------------


class TestBaseInferenceServer:
    def test_stop_sets_flag(self, server_config: ServerConfig) -> None:
        server = StubServer(server_config, [])
        assert not server.stopped()
        server.stop()
        assert server.stopped()

    def test_is_daemon_thread(self, server_config: ServerConfig) -> None:
        server = StubServer(server_config, [])
        assert server.daemon is True

    def test_config_stored(self, server_config: ServerConfig) -> None:
        server = StubServer(server_config, [])
        assert server.config is server_config
