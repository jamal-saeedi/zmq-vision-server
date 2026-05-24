"""Dataclass-based configuration for servers and clients."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    """Configuration for an inference server instance.

    All fields can be overridden via environment variables. The ``device``
    field accepts ``"auto"``, ``"cpu"``, or ``"cuda"``; ``"auto"`` selects
    CUDA when available and falls back to CPU.
    """

    host: str = field(default_factory=lambda: os.getenv("ZMQ_HOST", "*"))
    port: int = field(default_factory=lambda: int(os.getenv("ZMQ_PORT", "5576")))
    model_name: str = field(
        default_factory=lambda: os.getenv("ZMQ_MODEL", "mobilenetv2_100")
    )
    backend_endpoint: str = "inproc://backend"
    top_k: int = 3
    poll_timeout_ms: int = 5000
    device: str = field(
        default_factory=lambda: os.getenv("ZMQ_DEVICE", "auto")
    )  # auto|cpu|cuda

    @property
    def bind_address(self) -> str:
        """Return the full TCP bind address string."""
        return f"tcp://{self.host}:{self.port}"


@dataclass
class ClientConfig:
    """Configuration for an inference client instance.

    All fields can be overridden via environment variables.
    """

    host: str = field(
        default_factory=lambda: os.getenv("ZMQ_SERVER_HOST", "localhost")
    )
    port: int = field(default_factory=lambda: int(os.getenv("ZMQ_PORT", "5576")))
    timeout_ms: int = field(
        default_factory=lambda: int(os.getenv("ZMQ_TIMEOUT_MS", "30000"))
    )
    max_retries: int = 3

    @property
    def server_address(self) -> str:
        """Return the full TCP server address string."""
        return f"tcp://{self.host}:{self.port}"
