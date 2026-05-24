"""Synchronous polling ZMQ inference client with retry logic."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Union

import zmq
from PIL import Image

from zmq_inference.client.base import BaseInferenceClient
from zmq_inference.core.config import ClientConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse
from zmq_inference.utils.image import encode_image

logger = get_logger("client.sync")


class SyncClient(BaseInferenceClient):
    """Synchronous inference client with polling and automatic retry.

    Uses a ZMQ DEALER socket with a unique identity per instance. On timeout
    the client will retry up to ``config.max_retries`` times before raising
    a :class:`TimeoutError`.

    Example::

        from zmq_inference.client.sync_client import SyncClient

        with SyncClient() as client:
            response = client.infer("cat.jpg")
            for pred in response.predictions:
                print(pred)
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        super().__init__(config)
        self._context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        self._poller: zmq.Poller | None = None
        self._identity: str = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create ZMQ context and connect the DEALER socket to the server."""
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._identity)
        self._socket.connect(self.config.server_address)

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

        logger.info(
            "Connected to %s (identity=%s)", self.config.server_address, self._identity
        )

    def disconnect(self) -> None:
        """Close the socket and terminate the ZMQ context."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self._poller = None
        logger.info("Disconnected from %s", self.config.server_address)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        source: Union[str, Path, Image.Image],
    ) -> InferenceResponse:
        """Encode *source* and send it to the server, returning the response.

        Retries up to ``config.max_retries`` times when the server does not
        respond within ``config.timeout_ms`` milliseconds.

        Args:
            source: Path to an image file or a :class:`PIL.Image.Image`.

        Returns:
            Parsed :class:`~zmq_inference.core.protocol.InferenceResponse`.

        Raises:
            TimeoutError: When the server does not respond after all retries.
            RuntimeError: When :meth:`connect` has not been called.
        """
        if self._socket is None or self._poller is None:
            raise RuntimeError(
                "SyncClient is not connected. Call connect() or use as a context manager."
            )

        payload = encode_image(source)
        request = InferenceRequest(payload=payload)

        for attempt in range(1, self.config.max_retries + 1):
            logger.debug(
                "Sending request %s (attempt %d/%d)",
                request.request_id,
                attempt,
                self.config.max_retries,
            )
            self._socket.send_json(request.to_dict())

            events = dict(self._poller.poll(self.config.timeout_ms))
            if self._socket in events and events[self._socket] == zmq.POLLIN:
                raw: str = self._socket.recv_string()
                response = InferenceResponse.from_json(raw)
                logger.info(
                    "Received response for %s in %.1f ms",
                    response.request_id,
                    response.elapsed_ms,
                )
                return response

            logger.warning(
                "Timeout on attempt %d/%d (timeout_ms=%d)",
                attempt,
                self.config.max_retries,
                self.config.timeout_ms,
            )

        raise TimeoutError(
            f"Server at {self.config.server_address} did not respond after "
            f"{self.config.max_retries} attempt(s) "
            f"(timeout_ms={self.config.timeout_ms})."
        )
