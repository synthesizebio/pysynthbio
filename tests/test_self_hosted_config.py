"""Tests for self-hosted env-var resolution and optional-auth GET endpoints."""

from unittest import mock

import pysynthbio
from pysynthbio.http_client import (
    API_BASE_URL,
    env_flag,
    resolve_base_url,
    self_hosted_enabled,
)


def test_env_flag_truthy_values(monkeypatch):
    for value in ["1", "true", "TRUE", "yes", "on"]:
        monkeypatch.setenv("SYNTHESIZE_SELF_HOSTED", value)
        assert env_flag("SYNTHESIZE_SELF_HOSTED")
    for value in ["0", "false", "", "no"]:
        monkeypatch.setenv("SYNTHESIZE_SELF_HOSTED", value)
        assert not env_flag("SYNTHESIZE_SELF_HOSTED")


def test_resolve_base_url_precedence(monkeypatch):
    monkeypatch.delenv("SYNTHESIZE_API_BASE_URL", raising=False)
    assert resolve_base_url(None) == API_BASE_URL
    monkeypatch.setenv("SYNTHESIZE_API_BASE_URL", "http://box:8080")
    assert resolve_base_url(None) == "http://box:8080"
    # Explicit non-default arg wins over env
    assert resolve_base_url("http://explicit:9000") == "http://explicit:9000"


def test_self_hosted_enabled_precedence(monkeypatch):
    monkeypatch.setenv("SYNTHESIZE_SELF_HOSTED", "1")
    assert self_hosted_enabled(None) is True
    assert self_hosted_enabled(False) is False  # explicit wins


def test_get_example_query_self_hosted_no_token(monkeypatch):
    monkeypatch.delenv("SYNTHESIZE_API_KEY", raising=False)
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"example_query": {"modality": "bulk"}}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    with mock.patch("pysynthbio.http_client.requests.get", fake_get):
        result = pysynthbio.get_example_query(
            "gem-1-bulk", api_base_url="http://box:8080", self_hosted=True
        )

    assert result["example_query"]["modality"] == "bulk"
    assert captured["url"] == "http://box:8080/api/models/gem-1-bulk/example-query"
    assert "Authorization" not in captured["headers"]


def test_get_example_query_self_hosted_attaches_token_when_set(monkeypatch):
    monkeypatch.setenv("SYNTHESIZE_API_KEY", "sbio_test")

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"example_query": {}}

    captured = {}

    def fake_get(url, headers, timeout):
        captured["headers"] = headers
        return FakeResponse()

    with mock.patch("pysynthbio.http_client.requests.get", fake_get):
        pysynthbio.get_example_query("gem-1-bulk", self_hosted=True)

    assert captured["headers"]["Authorization"] == "Bearer sbio_test"
