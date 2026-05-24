"""
zmq_inference - ZMQ-based high-performance inference server and client.

Supports PyTorch (timm) and TensorFlow (Keras) backends for ImageNet classification
with a clean async/sync client API and a Click CLI.
"""

__version__ = "0.2.0"
__author__ = "Jamal Saeedi"
__email__ = "jamal.saeedi7@gmail.com"

from zmq_inference.core.config import ClientConfig, ServerConfig
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse, Prediction

__all__ = [
    "__version__",
    "ServerConfig",
    "ClientConfig",
    "InferenceRequest",
    "InferenceResponse",
    "Prediction",
]
