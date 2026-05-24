"""Tests for zmq_inference.client.sync_client.SyncClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
import zmq

from zmq_inference.client.sync_client import SyncClient
from zmq_inference.core.config import ClientConfig
from zmq_inference.core.protocol import InferenceResponse, Prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(request_id: str = "req-1") -> InferenceResponse:
    return InferenceResponse(
        request_id=request_id,
        predictions=[Prediction(label="tabby cat", confidence=0.95)],
        image="",
        elapsed_ms=10.0,
    )


def _mock_zmq(response: InferenceResponse, socket_mock: MagicMock) -> None:
    """Configure socket_mock so that poll returns POLLIN and recv_string returns JSON."""
    socket_mock.recv_string.return_value = response.to_json()


# ---------------------------------------------------------------------------
# Test SyncClient.connect / disconnect
# ---------------------------------------------------------------------------


class TestSyncClientLifecycle:
    def test_connect_creates_socket(self, client_config: ClientConfig) -> None:
        with patch("zmq_inference.client.sync_client.zmq.Context") as MockCtx:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            MockCtx.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            client = SyncClient(client_config)
            client.connect()

            mock_ctx.socket.assert_called_once_with(zmq.DEALER)
            mock_socket.setsockopt_string.assert_called_once()
            mock_socket.connect.assert_called_once_with(client_config.server_address)

    def test_disconnect_closes_socket_and_context(self, client_config: ClientConfig) -> None:
        with patch("zmq_inference.client.sync_client.zmq.Context") as MockCtx:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            MockCtx.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket

            client = SyncClient(client_config)
            client.connect()
            client.disconnect()

            mock_socket.close.assert_called_once_with(linger=0)
            mock_ctx.term.assert_called_once()

    def test_context_manager_calls_connect_and_disconnect(
        self, client_config: ClientConfig
    ) -> None:
        with patch("zmq_inference.client.sync_client.zmq.Context") as MockCtx:
            mock_ctx = MagicMock()
            mock_socket = MagicMock()
            MockCtx.return_value = mock_ctx
            mock_ctx.socket.return_value = mock_socket
            mock_socket.poll.return_value = {}

            with SyncClient(client_config) as _client:
                pass  # just check lifecycle

            mock_socket.close.assert_called_once()
            mock_ctx.term.assert_called_once()


# ---------------------------------------------------------------------------
# Test SyncClient.infer
# ---------------------------------------------------------------------------


class TestSyncClientInfer:
    def _setup(self, client_config: ClientConfig, response: InferenceResponse):
        """Return a (SyncClient, mock_socket, mock_poller) triple."""
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_socket.recv_string.return_value = response.to_json()

        # Poller returns POLLIN on first call
        mock_poller = MagicMock()
        mock_poller.poll.return_value = {mock_socket: zmq.POLLIN}

        client = SyncClient(client_config)
        client._context = mock_ctx
        client._socket = mock_socket
        client._poller = mock_poller
        return client, mock_socket, mock_poller

    def test_infer_returns_response(
        self, client_config: ClientConfig, tiny_image_path: str
    ) -> None:
        expected = _make_response("req-abc")
        client, mock_socket, _mock_poller = self._setup(client_config, expected)

        result = client.infer(tiny_image_path)

        assert isinstance(result, InferenceResponse)
        assert result.request_id == "req-abc"
        assert len(result.predictions) == 1
        assert result.predictions[0].label == "tabby cat"

    def test_infer_sends_json(
        self, client_config: ClientConfig, tiny_image_path: str
    ) -> None:
        expected = _make_response()
        client, mock_socket, _mock_poller = self._setup(client_config, expected)
        client.infer(tiny_image_path)

        mock_socket.send_json.assert_called_once()
        sent_dict = mock_socket.send_json.call_args[0][0]
        assert "payload" in sent_dict
        assert "request_id" in sent_dict

    def test_infer_raises_on_timeout(
        self, client_config: ClientConfig, tiny_image_path: str
    ) -> None:
        client_config.max_retries = 2
        mock_ctx = MagicMock()
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket

        mock_poller = MagicMock()
        # Always returns empty dict → always times out.
        mock_poller.poll.return_value = {}

        client = SyncClient(client_config)
        client._context = mock_ctx
        client._socket = mock_socket
        client._poller = mock_poller

        with pytest.raises(TimeoutError):
            client.infer(tiny_image_path)

        # Should have tried max_retries times.
        assert mock_poller.poll.call_count == client_config.max_retries

    def test_infer_not_connected_raises_runtime_error(
        self, client_config: ClientConfig, tiny_image_path: str
    ) -> None:
        client = SyncClient(client_config)
        with pytest.raises(RuntimeError, match="not connected"):
            client.infer(tiny_image_path)

    def test_infer_accepts_pil_image(
        self, client_config: ClientConfig, tiny_pil_image: object
    ) -> None:
        expected = _make_response()
        from PIL import Image as PILImage

        assert isinstance(tiny_pil_image, PILImage.Image)
        client, mock_socket, _mock_poller = self._setup(client_config, expected)
        result = client.infer(tiny_pil_image)  # type: ignore[arg-type]
        assert result.ok
