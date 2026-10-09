"""A cold model's wait limit is transport configuration, not model input."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from pysynthbio import call_model_api as api


@pytest.mark.parametrize("configured, expected", [(None, 600), (1800, 1800)])
def test_prediction_passes_wait_limit_to_http_not_request_body(
    monkeypatch, configured, expected
):
    captured = {}

    @contextmanager
    def stream(**kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(content=b"complete response")

    monkeypatch.setattr(api, "open_arrow_stream", stream)
    options = {} if configured is None else {"self_hosted_timeout_seconds": configured}
    response = api.predict_query(
        {"inputs": [], "options": {}},
        "gem2-preview",
        api_base_url="http://localhost:9000",
        self_hosted=True,
        raw_response=True,
        **options,
    )
    assert response == b"complete response"
    assert captured["timeout"] == expected
    assert captured["json"] == {"inputs": [], "options": {}, "source": "pysynthbio"}


@pytest.mark.parametrize(
    "value", [None, False, 0, -1, float("inf"), float("nan"), "1800"]
)
def test_invalid_wait_limit_never_starts_http(monkeypatch, value):
    def unexpected(**kwargs):
        pytest.fail("Invalid timeout reached HTTP")

    monkeypatch.setattr(api, "open_arrow_stream", unexpected)
    with pytest.raises(ValueError, match="positive finite"):
        api.predict_query(
            {}, "gem2-preview", self_hosted=True, self_hosted_timeout_seconds=value
        )
