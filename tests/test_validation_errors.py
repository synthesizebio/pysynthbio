"""Presentation and lossless access to Platform validation responses."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from pysynthbio.call_model_api import predict_query
from pysynthbio.http_client import ValidationError, api_request


def rejected_response(body, status=422):
    response = Mock(status_code=status, text=body)
    response.raise_for_status.side_effect = requests.HTTPError(response=response)
    return response


def test_production_validation_through_predict_query(monkeypatch):
    monkeypatch.setenv("SYNTHESIZE_API_KEY", "test-token")
    body = (Path(__file__).parent / "fixtures/gem2-validation.json").read_text()
    payload = json.loads(body)
    with patch(
        "pysynthbio.http_client.requests.request",
        return_value=rejected_response(body),
    ) as request, patch("pysynthbio.call_model_api._poll_model_query") as poll:
        with pytest.raises(ValidationError) as caught:
            predict_query({"inputs": []}, model_id="gem-2-sc", auto_authenticate=False)
    error = caught.value
    assert error.status_code == 422
    assert error.response_body == body
    assert error.response_json == payload
    assert str(error) == "Invalid GEM-2 request (422):\n" + "\n".join(
        f"- {issue['path']}: {issue['message']}" for issue in payload["errors"]
    )
    assert isinstance(error.__cause__, requests.HTTPError)
    request.assert_called_once()
    poll.assert_not_called()


@pytest.mark.parametrize("status", [400, 422])
@pytest.mark.parametrize("path", ["inputs.items[1].metadata.age_years", ""])
def test_single_error_and_root_path(monkeypatch, status, path):
    monkeypatch.setenv("SYNTHESIZE_API_KEY", "test-token")
    payload = {"errors": [{"path": path, "message": "Correct the value.", "code": "x"}]}
    body = json.dumps(payload)
    with patch(
        "pysynthbio.http_client.requests.request",
        return_value=rejected_response(body, status),
    ), pytest.raises(ValidationError) as caught:
        api_request("POST", "/api/models/gem-2/predict", json={})
    assert str(caught.value) == (
        f"Invalid request ({status}):\n- {path or '(request)'}: Correct the value."
    )
    assert caught.value.response_json == payload


@pytest.mark.parametrize(
    "body",
    [
        "Legacy validation error",
        "{broken json",
        "null",
        "[]",
        '{"errors": []}',
        '{"errors": {"path": "x", "message": "bad"}}',
        '{"errors": [null]}',
        '{"errors": [{"path": 1, "message": "bad"}]}',
        '{"errors": [{"path": "x", "message": ["bad"]}]}',
        '{"errors": [{"path": "x", "message": " "}]}',
        '{"errors": [{"path": "x", "message": "bad"}, {}]}',
    ],
)
def test_fallback_preserves_all_response_data(monkeypatch, body):
    monkeypatch.setenv("SYNTHESIZE_API_KEY", "test-token")
    with patch(
        "pysynthbio.http_client.requests.request",
        return_value=rejected_response(body),
    ), pytest.raises(ValidationError) as caught:
        api_request("POST", "/api/models/gem-1-bulk/predict", json={})
    error = caught.value
    assert str(error) == f"Invalid request (422): {body}"
    assert error.response_body == body
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    assert error.response_json == parsed


def test_existing_constructor_remains_compatible():
    error = ValidationError("Original message", 400)
    assert str(error) == "Original message"
    assert error.status_code == 400
    assert error.response_body is None
    assert error.response_json is None
