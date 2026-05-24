"""Tests for zmq_inference.core.protocol serialisation."""

from __future__ import annotations

import json

import pytest

from zmq_inference.core.protocol import InferenceRequest, InferenceResponse, Prediction


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_to_dict(self, sample_prediction: Prediction) -> None:
        d = sample_prediction.to_dict()
        assert d["label"] == "tabby cat"
        assert abs(d["confidence"] - 0.92345) < 1e-6

    def test_from_dict_roundtrip(self, sample_prediction: Prediction) -> None:
        restored = Prediction.from_dict(sample_prediction.to_dict())
        assert restored.label == sample_prediction.label
        assert abs(restored.confidence - sample_prediction.confidence) < 1e-6

    def test_str(self, sample_prediction: Prediction) -> None:
        s = str(sample_prediction)
        assert "tabby cat" in s
        assert "0.92345" in s


# ---------------------------------------------------------------------------
# InferenceRequest
# ---------------------------------------------------------------------------


class TestInferenceRequest:
    def test_auto_request_id(self) -> None:
        r1 = InferenceRequest(payload="abc")
        r2 = InferenceRequest(payload="abc")
        assert r1.request_id != r2.request_id

    def test_explicit_request_id(self) -> None:
        req = InferenceRequest(payload="abc", request_id="my-id")
        assert req.request_id == "my-id"

    def test_to_dict_keys(self, sample_request: InferenceRequest) -> None:
        d = sample_request.to_dict()
        assert set(d.keys()) == {"payload", "request_id", "metadata"}

    def test_from_dict_roundtrip(self, sample_request: InferenceRequest) -> None:
        restored = InferenceRequest.from_dict(sample_request.to_dict())
        assert restored.payload == sample_request.payload
        assert restored.request_id == sample_request.request_id
        assert restored.metadata == sample_request.metadata

    def test_to_json_is_valid_json(self, sample_request: InferenceRequest) -> None:
        raw = sample_request.to_json()
        parsed = json.loads(raw)
        assert parsed["request_id"] == sample_request.request_id

    def test_from_json_roundtrip(self, sample_request: InferenceRequest) -> None:
        restored = InferenceRequest.from_json(sample_request.to_json())
        assert restored.request_id == sample_request.request_id
        assert restored.payload == sample_request.payload

    def test_metadata_default_is_empty_dict(self) -> None:
        req = InferenceRequest(payload="x")
        assert req.metadata == {}

    def test_from_dict_missing_request_id_generates_new(self) -> None:
        req = InferenceRequest.from_dict({"payload": "abc"})
        assert req.request_id  # non-empty
        assert req.request_id != "abc"


# ---------------------------------------------------------------------------
# InferenceResponse
# ---------------------------------------------------------------------------


class TestInferenceResponse:
    def test_ok_property_no_error(self, sample_response: InferenceResponse) -> None:
        assert sample_response.ok is True

    def test_ok_property_with_error(self) -> None:
        resp = InferenceResponse(
            request_id="x",
            predictions=[],
            image="",
            elapsed_ms=0.0,
            error="something went wrong",
        )
        assert resp.ok is False

    def test_to_dict_roundtrip(self, sample_response: InferenceResponse) -> None:
        d = sample_response.to_dict()
        restored = InferenceResponse.from_dict(d)
        assert restored.request_id == sample_response.request_id
        assert restored.elapsed_ms == sample_response.elapsed_ms
        assert len(restored.predictions) == len(sample_response.predictions)
        assert restored.predictions[0].label == sample_response.predictions[0].label

    def test_to_json_roundtrip(self, sample_response: InferenceResponse) -> None:
        raw = sample_response.to_json()
        restored = InferenceResponse.from_json(raw)
        assert restored.request_id == sample_response.request_id
        assert restored.ok is True

    def test_str_no_error(self, sample_response: InferenceResponse) -> None:
        s = str(sample_response)
        assert "InferenceResponse" in s
        assert "42.5" in s

    def test_str_with_error(self) -> None:
        resp = InferenceResponse(
            request_id="y", predictions=[], image="", elapsed_ms=0.0, error="boom"
        )
        assert "boom" in str(resp)

    def test_from_dict_defaults(self) -> None:
        resp = InferenceResponse.from_dict(
            {"request_id": "z", "predictions": [], "image": "", "elapsed_ms": 0}
        )
        assert resp.error is None
        assert resp.ok is True
