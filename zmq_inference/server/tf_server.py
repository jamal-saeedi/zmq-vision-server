"""TensorFlow/Keras inference server for ImageNet classification."""

from __future__ import annotations

import os
from typing import Any, List

import numpy as np
from PIL import Image

from zmq_inference.core.config import ServerConfig
from zmq_inference.core.logging import get_logger
from zmq_inference.core.protocol import Prediction
from zmq_inference.server.base import BaseInferenceServer

logger = get_logger("server.tf")


class TFInferenceServer(BaseInferenceServer):
    """TensorFlow/Keras-backed inference server for ImageNet classification.

    Uses ``keras.applications.MobileNet`` by default. The model is loaded
    once in :meth:`load_model` and kept in memory for the server lifetime.
    Inference uses :func:`model.predict` wrapped in a ``tf.function``-compiled
    forward pass for efficient GPU utilisation.

    Args:
        config: Server configuration. ``config.model_name`` is informational
                for TF (the Keras model is fixed to MobileNet by default).

    Example::

        from zmq_inference import ServerConfig
        from zmq_inference.server.tf_server import TFInferenceServer

        server = TFInferenceServer(ServerConfig(port=5576))
        server.start()
    """

    def __init__(self, config: ServerConfig | None = None) -> None:
        super().__init__(config or ServerConfig())
        self._model = None
        self._preprocess_input = None
        self._decode_predictions = None

    # ------------------------------------------------------------------
    # BaseInferenceServer interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the Keras MobileNet model and configure GPU memory growth."""
        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet import (  # type: ignore[import]
                MobileNet,
                decode_predictions,
                preprocess_input,
            )
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. "
                "Install with: pip install zmq-inference[tf]"
            ) from exc

        # Enable memory growth to avoid pre-allocating all GPU memory.
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass  # Must be set before GPUs are initialised.
            logger.info("GPU(s) detected: %d device(s), memory growth enabled.", len(gpus))
        else:
            logger.info("No GPU detected. Running on CPU.")

        # Limit inter/intra-op parallelism to the available CPU count.
        max_workers = os.cpu_count() or 1
        tf.config.threading.set_inter_op_parallelism_threads(max_workers)
        tf.config.threading.set_intra_op_parallelism_threads(max_workers)

        self._model = MobileNet(weights="imagenet")
        self._preprocess_input = preprocess_input
        self._decode_predictions = decode_predictions

        # Warm-up: compile the graph with a dummy forward pass.
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        self._model.predict(dummy, verbose=0)
        logger.info("Keras MobileNet loaded and warm-up complete.")

    def preprocess(self, img: Image.Image) -> Any:
        """Resize and normalise a PIL image for MobileNet.

        Args:
            img: Input PIL image.

        Returns:
            A ``(1, 224, 224, 3)`` float32 NumPy array.
        """
        img_rgb = img.convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img_rgb, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return self._preprocess_input(arr)  # type: ignore[misc]

    def predict(self, tensor: Any) -> List[Prediction]:
        """Run inference and return top-k decoded ImageNet predictions.

        Args:
            tensor: Preprocessed float32 NumPy array.

        Returns:
            List of :class:`Prediction` objects sorted by confidence (desc).
        """
        raw_preds = self._model.predict(tensor, verbose=0)  # type: ignore[union-attr]
        decoded = self._decode_predictions(raw_preds, top=self.config.top_k)[0]  # type: ignore[misc]

        predictions = [
            Prediction(
                label=str(label),
                confidence=round(float(prob), 6),
            )
            for _, label, prob in decoded
        ]
        return predictions
