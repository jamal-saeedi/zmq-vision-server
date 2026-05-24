"""Abstract base class for inference clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from PIL import Image

from zmq_inference.core.config import ClientConfig
from zmq_inference.core.protocol import InferenceResponse


class BaseInferenceClient(ABC):
    """Abstract base class for ZMQ inference clients.

    Concrete subclasses (synchronous or async) must implement :meth:`connect`,
    :meth:`disconnect`, and :meth:`infer`.

    The base class supports the context-manager protocol so that sockets are
    always closed even when an exception occurs::

        with SyncClient() as client:
            response = client.infer("image.jpg")
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config: ClientConfig = config or ClientConfig()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Open the ZMQ socket and connect to the server."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the ZMQ socket and clean up resources."""

    @abstractmethod
    def infer(
        self,
        source: Union[str, Path, Image.Image],
    ) -> InferenceResponse:
        """Send an image to the server and return the parsed response.

        Args:
            source: A filesystem path or :class:`PIL.Image.Image`.

        Returns:
            An :class:`~zmq_inference.core.protocol.InferenceResponse`.
        """

    # ------------------------------------------------------------------
    # Context manager helpers (sync)
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseInferenceClient":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()
