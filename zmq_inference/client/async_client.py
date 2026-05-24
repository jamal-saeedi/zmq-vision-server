"""Async ZMQ inference client using asyncio."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Union

import zmq
import zmq.asyncio
from PIL import Image

from zmq_inference.core.config import ClientConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse
from zmq_inference.utils.image import encode_image

logger = get_logger("client.async")


class AsyncClient:
    """Asynchronous ZMQ inference client.

    Uses ``zmq.asyncio`` to avoid blocking the event loop during network I/O.
    Supports the async context-manager protocol::

        async with AsyncClient() as client:
            response = await client.infer("cat.jpg")
            for pred in response.predictions:
                print(pred)

    Args:
        config: Optional client configuration. Defaults to :class:`ClientConfig`
                with values taken from environment variables.
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config: ClientConfig = config or ClientConfig()
        self._context: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._identity: str = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open an async ZMQ DEALER socket and connect to the server."""
        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt_string(zmq.IDENTITY, self._identity)
        self._socket.connect(self.config.server_address)
        logger.info(
            "Async client connected to %s (identity=%s)",
            self.config.server_address,
            self._identity,
        )

    async def disconnect(self) -> None:
        """Close the socket and terminate the ZMQ context."""
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        logger.info("Async client disconnected from %s", self.config.server_address)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    async def infer(
        self,
        source: Union[str, Path, Image.Image],
    ) -> InferenceResponse:
        """Encode *source*, send it to the server, and await the response.

        Args:
            source: Path to an image file or a :class:`PIL.Image.Image`.

        Returns:
            Parsed :class:`~zmq_inference.core.protocol.InferenceResponse`.

        Raises:
            TimeoutError:  When the server does not respond within
                           ``config.timeout_ms`` milliseconds.
            RuntimeError:  When :meth:`connect` has not been called.
        """
        if self._socket is None:
            raise RuntimeError(
                "AsyncClient is not connected. "
                "Call await connect() or use as an async context manager."
            )

        payload = encode_image(source)
        request = InferenceRequest(payload=payload)

        logger.debug("Sending async request %s", request.request_id)
        await self._socket.send_json(request.to_dict())

        timeout_s = self.config.timeout_ms / 1000.0
        try:
            raw: str = await asyncio.wait_for(
                self._socket.recv_string(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Server at {self.config.server_address} did not respond within "
                f"{self.config.timeout_ms} ms."
            )

        response = InferenceResponse.from_json(raw)
        logger.info(
            "Received async response for %s in %.1f ms",
            response.request_id,
            response.elapsed_ms,
        )
        return response

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AsyncClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()
