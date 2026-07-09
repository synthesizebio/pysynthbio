"""Tests for self-hosted env-var resolution and optional-auth GET endpoints."""

from unittest import mock

import pytest

import pysynthbio
from pysynthbio.http_client import (
    API_BASE_URL,
    configure,
    env_flag,
    legacy_per_model_env_var,
    per_model_env_var,
    resolve_base_url,
    self_hosted_enabled,
)


@pytest.fixture(autouse=True)
def reset_config(monkeypatch):
    configure(reset=True)
    monkeypatch.delenv("SYNTHESIZE_CONFIG", raising=False)
    monkeypatch.delenv("SYNTHESIZE_SELF_HOSTED", raising=False)
    monkeypatch.delenv("SYNTHESIZE_API_BASE_URL", raising=False)
    monkeypatch.delenv("SYNTHESIZE_ENDPOINT_GEM_1_BULK", raising=False)
    monkeypatch.delenv("SYNTHESIZE_ENDPOINT_GEM_1_SC", raising=False)
    monkeypatch.delenv("SYNTHESIZE_ENDPOINT_GEM_2", raising=False)
    monkeypatch.delenv("SYNTHESIZE_API_BASE_URL__GEM_1_BULK", raising=False)
    monkeypatch.delenv("SYNTHESIZE_API_BASE_URL__GEM_1_SC", raising=False)
    yield
    configure(reset=True)


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


def test_per_model_env_var_naming():
    assert per_model_env_var("gem-1-bulk") == "SYNTHESIZE_ENDPOINT_GEM_1_BULK"
    assert per_model_env_var("gem-1-sc") == "SYNTHESIZE_ENDPOINT_GEM_1_SC"
    assert per_model_env_var("gem-2") == "SYNTHESIZE_ENDPOINT_GEM_2"
    # Variant slugs normalize to their base model's variable.
    assert (
        per_model_env_var("gem-1-bulk_predict-metadata")
        == "SYNTHESIZE_ENDPOINT_GEM_1_BULK"
    )
    assert (
        per_model_env_var("gem-1-sc_reference-conditioning")
        == "SYNTHESIZE_ENDPOINT_GEM_1_SC"
    )


def test_resolve_base_url_per_model(monkeypatch):
    monkeypatch.setenv("SYNTHESIZE_ENDPOINT_GEM_1_BULK", "http://bulk:8080/")
    monkeypatch.setenv("SYNTHESIZE_ENDPOINT_GEM_1_SC", "http://sc:8080")

    # Each model resolves to its own host, no per-call URL needed.
    assert resolve_base_url(model_id="gem-1-bulk") == "http://bulk:8080"
    assert resolve_base_url(model_id="gem-1-sc") == "http://sc:8080"
    # Variant slugs share the base model's host.
    assert (
        resolve_base_url(model_id="gem-1-bulk_predict-metadata") == "http://bulk:8080"
    )
    assert (
        resolve_base_url(model_id="gem-1-sc_reference-conditioning") == "http://sc:8080"
    )
    # Explicit arg still wins over the per-model env var.
    assert (
        resolve_base_url("http://explicit:9000", model_id="gem-1-bulk")
        == "http://explicit:9000"
    )
    # A model with no per-model var falls back to the global, then the default.
    assert resolve_base_url(model_id="gem-1-unknown") == API_BASE_URL
    monkeypatch.setenv("SYNTHESIZE_API_BASE_URL", "http://global:8080")
    assert resolve_base_url(model_id="gem-1-unknown") == "http://global:8080"


def test_resolve_base_url_per_model_beats_global(monkeypatch):
    monkeypatch.setenv("SYNTHESIZE_API_BASE_URL", "http://global:8080")
    monkeypatch.setenv("SYNTHESIZE_ENDPOINT_GEM_1_SC", "http://sc:8080")
    # Per-model wins over global when present...
    assert resolve_base_url(model_id="gem-1-sc") == "http://sc:8080"
    # ...but a model without its own var uses the global.
    assert resolve_base_url(model_id="gem-1-bulk") == "http://global:8080"


def test_legacy_per_model_env_var_is_still_supported(monkeypatch):
    assert (
        legacy_per_model_env_var("gem-1-bulk") == "SYNTHESIZE_API_BASE_URL__GEM_1_BULK"
    )
    monkeypatch.setenv("SYNTHESIZE_API_BASE_URL__GEM_1_BULK", "http://legacy:8080")
    assert resolve_base_url(model_id="gem-1-bulk") == "http://legacy:8080"


def test_configure_precedence(monkeypatch):
    monkeypatch.setenv("SYNTHESIZE_ENDPOINT_GEM_1_BULK", "http://env-bulk:8080")
    monkeypatch.setenv("SYNTHESIZE_API_BASE_URL", "http://env-global:8080")

    configure(
        api_base_url="http://configured-global:8080",
        endpoint_gem_1_bulk="http://configured-bulk:8080",
        model_endpoints={"gem-1-sc": "http://configured-sc:8080"},
        self_hosted=True,
    )

    assert resolve_base_url(model_id="gem-1-bulk") == "http://configured-bulk:8080"
    assert resolve_base_url(model_id="gem-1-sc") == "http://configured-sc:8080"
    assert resolve_base_url(model_id="gem-2") == "http://configured-global:8080"
    assert (
        resolve_base_url("http://per-call:8080", model_id="gem-1-bulk")
        == "http://per-call:8080"
    )
    assert self_hosted_enabled(None) is True


def test_config_file_resolution(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("""
api_base_url = "http://file-global:8080"
self_hosted = true

[model_endpoints]
gem_1_bulk = "http://file-bulk:8080"
"gem-1-sc" = "http://file-sc:8080"
""".strip())
    monkeypatch.setenv("SYNTHESIZE_CONFIG", str(config_path))

    assert resolve_base_url(model_id="gem-1-bulk") == "http://file-bulk:8080"
    assert resolve_base_url(model_id="gem-1-sc") == "http://file-sc:8080"
    assert resolve_base_url(model_id="gem-2") == "http://file-global:8080"
    assert self_hosted_enabled(None) is True


def test_env_beats_config_file(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text('api_base_url = "http://file-global:8080"\n')
    monkeypatch.setenv("SYNTHESIZE_CONFIG", str(config_path))
    monkeypatch.setenv("SYNTHESIZE_API_BASE_URL", "http://env-global:8080")

    assert resolve_base_url(model_id="gem-2") == "http://env-global:8080"


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
