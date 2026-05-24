"""Abstract base inference server using ZMQ ROUTER/DEALER pattern."""

from __future__ import annotations

import io
import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, List

import zmq
from PIL import Image

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse, Prediction
from zmq_inference.utils.image import decode_image, encode_image

logger = get_logger("server.base")


class RequestHandler(threading.Thread):
    """Worker thread that processes a single inference request.

    Each incoming request spawns one :class:`RequestHandler` which connects
    to the inproc DEALER backend and sends the response back through it.

    Args:
        context:  Shared :class:`zmq.Context`.
        client_id: Raw identity frame from the ROUTER frontend.
        raw_msg:  Parsed JSON dict from the client.
        server:   Parent server instance providing ``predict`` / ``preprocess``.
    """

    def __init__(
        self,
        context: zmq.Context,
        client_id: bytes,
        raw_msg: dict,
        server: "BaseInferenceServer",
    ) -> None:
        super().__init__(daemon=True)
        self.context = context
        self.client_id = client_id
        self.raw_msg = raw_msg
        self.server = server

    def run(self) -> None:
        worker: zmq.Socket = self.context.socket(zmq.DEALER)
        worker.connect(self.server.config.backend_endpoint)

        t0 = time.perf_counter()
        response: InferenceResponse

        try:
            request = InferenceRequest.from_dict(self.raw_msg)
            img: Image.Image = decode_image(request.payload)
            tensor = self.server.preprocess(img)
            predictions: List[Prediction] = self.server.predict(tensor)

            # Echo back the (possibly resized) original image.
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG")
            import base64
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            elapsed_ms = (time.perf_counter() - t0) * 1000
            response = InferenceResponse(
                request_id=request.request_id,
                predictions=predictions,
                image=image_b64,
                elapsed_ms=elapsed_ms,
            )
            logger.info(
                "request=%s elapsed_ms=%.1f predictions=%s",
                request.request_id,
                elapsed_ms,
                [str(p) for p in predictions],
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.exception("Error processing request: %s", exc)
            response = InferenceResponse(
                request_id=self.raw_msg.get("request_id", "unknown"),
                predictions=[],
                image="",
                elapsed_ms=elapsed_ms,
                error=str(exc),
            )

        worker.send(self.client_id, zmq.SNDMORE)
        worker.send_string(response.to_json())
        worker.close()


class BaseInferenceServer(ABC, threading.Thread):
    """Abstract base class for ZMQ ROUTER/DEALER inference servers.

    Subclasses must implement :meth:`load_model`, :meth:`preprocess`, and
    :meth:`predict`. The network loop is provided by this class.

    Usage::

        server = TorchInferenceServer(ServerConfig(port=5576))
        server.start()
        # ... later ...
        server.stop()
        server.join()
    """

    def __init__(self, config: ServerConfig) -> None:
        super().__init__(daemon=True)
        self.config = config
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory.

        Called once before the event loop starts. Implementations should
        store the model as an instance attribute.
        """

    @abstractmethod
    def preprocess(self, img: Image.Image) -> Any:
        """Transform a PIL image into the tensor expected by the model.

        Args:
            img: RGB PIL image.

        Returns:
            A backend-specific tensor (``torch.Tensor`` or ``np.ndarray``).
        """

    @abstractmethod
    def predict(self, tensor: Any) -> List[Prediction]:
        """Run inference on a preprocessed tensor.

        Args:
            tensor: Output of :meth:`preprocess`.

        Returns:
            Ordered list of :class:`Prediction` objects (highest confidence first).
        """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the server to shut down after the current poll cycle."""
        self._stop_event.set()

    def stopped(self) -> bool:
        """``True`` when the stop signal has been sent."""
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # ZMQ event loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the ROUTER/DEALER proxy loop (called by :meth:`threading.Thread.start`)."""
        logger.info("Loading model (%s)…", self.config.model_name)
        self.load_model()
        logger.info("Model loaded. Starting ZMQ server on %s", self.config.bind_address)

        context = zmq.Context()
        frontend: zmq.Socket = context.socket(zmq.ROUTER)
        frontend.bind(self.config.bind_address)

        backend: zmq.Socket = context.socket(zmq.DEALER)
        backend.bind(self.config.backend_endpoint)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        logger.info("Server ready and listening.")

        try:
            while not self.stopped():
                try:
                    events = dict(poller.poll(self.config.poll_timeout_ms))
                except zmq.ZMQError as exc:
                    logger.error("ZMQ poller error: %s", exc)
                    break

                if frontend in events and events[frontend] == zmq.POLLIN:
                    client_id: bytes = frontend.recv()
                    raw_msg: dict = frontend.recv_json()

                    handler = RequestHandler(context, client_id, raw_msg, self)
                    handler.start()

                if backend in events and events[backend] == zmq.POLLIN:
                    client_id = backend.recv()
                    msg: bytes = backend.recv()
                    frontend.send(client_id, zmq.SNDMORE)
                    frontend.send(msg)

        finally:
            logger.info("Shutting down ZMQ server…")
            poller.unregister(frontend)
            poller.unregister(backend)
            frontend.close(linger=0)
            backend.close(linger=0)
            context.term()
            logger.info("Server shut down cleanly.")
