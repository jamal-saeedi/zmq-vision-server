"""Request/Response dataclasses and JSON serialization for the ZMQ protocol."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class InferenceRequest:
    """A single inference request carrying a base64-encoded image.

    Attributes:
        payload:    Base64-encoded image string.
        request_id: Auto-generated UUID identifying the request.
        metadata:   Arbitrary caller-supplied key-value pairs.
    """

    payload: str  # base64 encoded image
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict suitable for JSON encoding."""
        return {
            "payload": self.payload,
            "request_id": self.request_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceRequest":
        """Deserialise from a plain dict."""
        return cls(
            payload=data["payload"],
            request_id=data.get("request_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "InferenceRequest":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(raw))


@dataclass
class Prediction:
    """A single top-k class prediction.

    Attributes:
        label:      Human-readable class label (e.g. ``"golden retriever"``).
        confidence: Model confidence in the range [0, 1].
    """

    label: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Prediction":
        return cls(label=data["label"], confidence=float(data["confidence"]))

    def __str__(self) -> str:
        return f"{self.label}: {self.confidence:.5f}"


@dataclass
class InferenceResponse:
    """A server response carrying predictions for a single request.

    Attributes:
        request_id:  Echoed from the originating :class:`InferenceRequest`.
        predictions: Ordered list of top-k :class:`Prediction` objects.
        image:       Base64-encoded result/echo image.
        elapsed_ms:  Server-side processing time in milliseconds.
        error:       Non-``None`` when the server encountered an error.
    """

    request_id: str
    predictions: List[Prediction]
    image: str  # base64 encoded result image
    elapsed_ms: float
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "predictions": [p.to_dict() for p in self.predictions],
            "image": self.image,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceResponse":
        return cls(
            request_id=data["request_id"],
            predictions=[Prediction.from_dict(p) for p in data.get("predictions", [])],
            image=data.get("image", ""),
            elapsed_ms=float(data.get("elapsed_ms", 0.0)),
            error=data.get("error"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "InferenceResponse":
        return cls.from_dict(json.loads(raw))

    @property
    def ok(self) -> bool:
        """``True`` when no error was reported."""
        return self.error is None

    def __str__(self) -> str:
        if self.error:
            return f"InferenceResponse(error={self.error!r})"
        preds = ", ".join(str(p) for p in self.predictions)
        return f"InferenceResponse(predictions=[{preds}], elapsed_ms={self.elapsed_ms:.1f})"
