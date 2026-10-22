from unittest.mock import Mock, patch

import pytest

from pysynthbio.verdict import VerdictAPIError, VerdictClient


def response(body, status=200):
    return Mock(status_code=status, json=Mock(return_value=body))


def test_model_submission_preserves_results_and_key():
    client = VerdictClient("vdt_test")
    output = {"result": {"unsupported": True, "release": "pilot"}}
    with patch(
        "pysynthbio.verdict.requests.request", return_value=response(output)
    ) as send:
        assert (
            client.predict("safety", "predict", {"inputs": []}, idempotency_key="same")
            == output
        )
        assert send.call_args.kwargs["headers"]["Idempotency-Key"] == "same"
        assert send.call_args.kwargs["allow_redirects"] is False
        assert (
            send.call_args.args[1]
            == "https://verdict.synthesize.bio/api/v1/models/safety/predict"
        )


def test_validation_errors_remain_structured():
    details = {"error": "invalid_model_input", "errors": [{"path": "/inputs/0"}]}
    with patch(
        "pysynthbio.verdict.requests.request", return_value=response(details, 422)
    ):
        with pytest.raises(VerdictAPIError) as failure:
            VerdictClient("key").models()
        assert failure.value.details == details
        assert failure.value.status_code == 422


def test_polling_never_resubmits():
    client = VerdictClient("key")
    with patch.object(
        client, "run", side_effect=[{"status": "running"}, {"status": "failed"}]
    ) as poll:
        with patch("pysynthbio.verdict.time.sleep"):
            assert client.wait("saved-id")["status"] == "failed"
        assert poll.call_count == 2


def test_endpoint_never_allows_plaintext_or_embedded_credentials():
    for origin in ["http://verdict.synthesize.bio", "https://user:secret@example.org"]:
        with pytest.raises(ValueError):
            VerdictClient("key", origin)
