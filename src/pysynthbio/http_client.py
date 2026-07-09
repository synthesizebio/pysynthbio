"""Centralized HTTP client with consistent error handling for the Synthesize Bio API."""

import os
import re
from pathlib import Path
from typing import Any, Optional

import requests

from pysynthbio.key_handlers import has_synthesize_token

API_BASE_URL = "https://app.synthesize.bio"
DEFAULT_TIMEOUT = 30

# Self-hosted predictions run synchronously on the partner's GPU box and can
# take minutes for large sample counts, so they use a much longer timeout than
# the (quick) hosted control-plane calls.
SELF_HOSTED_TIMEOUT = 600

ARROW_STREAM_CONTENT_TYPE = "application/vnd.apache.arrow.stream"

# Env vars that let a partner's data scientists point any client call at a
# self-hosted container without code changes.
SELF_HOSTED_ENV = "SYNTHESIZE_SELF_HOSTED"
API_BASE_URL_ENV = "SYNTHESIZE_API_BASE_URL"
CONFIG_FILE_ENV = "SYNTHESIZE_CONFIG"
DEFAULT_CONFIG_PATH = "~/.config/synthbio/config.toml"

_MODEL_ENDPOINT_ENVS = {
    "gem-1-bulk": "SYNTHESIZE_ENDPOINT_GEM_1_BULK",
    "gem-1-sc": "SYNTHESIZE_ENDPOINT_GEM_1_SC",
    "gem-2": "SYNTHESIZE_ENDPOINT_GEM_2",
}

_CONFIG: dict[str, Any] = {
    "api_base_url": None,
    "self_hosted": None,
    "model_endpoints": {},
}


def env_flag(name: str) -> bool:
    """Interpret an env var as a boolean (1/true/yes/on, case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# A model's variant slugs (reference-conditioning, predict-metadata) are served
# by the same container as their base model, so they resolve to the same host.
_MODEL_ID_SUFFIXES = ("_reference-conditioning", "_predict-metadata")


def _base_model_id(model_id: str) -> str:
    """Reduce a model slug to the base model that backs it (variants share a host)."""
    for suffix in _MODEL_ID_SUFFIXES:
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)]
    return model_id


def per_model_env_var(model_id: str) -> str:
    """Env var holding the self-hosted base URL for a specific model.

    The base model and all its variants map to one variable, e.g.
    ``gem-1-bulk`` and ``gem-1-bulk_predict-metadata`` both resolve to
    ``SYNTHESIZE_ENDPOINT_GEM_1_BULK``.
    """
    base_model_id = _base_model_id(model_id)
    if base_model_id in _MODEL_ENDPOINT_ENVS:
        return _MODEL_ENDPOINT_ENVS[base_model_id]
    key = re.sub(r"[^A-Z0-9]+", "_", base_model_id.upper())
    return f"SYNTHESIZE_ENDPOINT_{key}"


def legacy_per_model_env_var(model_id: str) -> str:
    """Previous per-model env var spelling kept for backward compatibility."""
    key = re.sub(r"[^A-Z0-9]+", "_", _base_model_id(model_id).upper())
    return f"{API_BASE_URL_ENV}__{key}"


def _model_config_key(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _base_model_id(model_id).lower()).strip("_")


def _normalize_base_url(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.rstrip("/")


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
        import tomli as tomllib

    with path.open("rb") as handle:
        return tomllib.load(handle)


def _config_file_path() -> Path:
    return Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_PATH)).expanduser()


def _file_config() -> dict:
    return _load_toml(_config_file_path())


def _configured_endpoint(model_id: str) -> Optional[str]:
    return _normalize_base_url(_CONFIG["model_endpoints"].get(_base_model_id(model_id)))


def _env_endpoint(model_id: str) -> Optional[str]:
    return _normalize_base_url(
        os.environ.get(per_model_env_var(model_id))
        or os.environ.get(legacy_per_model_env_var(model_id))
    )


def _file_endpoint(config: dict, model_id: str) -> Optional[str]:
    base_model_id = _base_model_id(model_id)
    key = _model_config_key(model_id)
    candidates = (
        f"endpoint_{key}",
        f"endpoint_{key.upper()}",
        base_model_id,
        key,
        key.upper(),
    )

    for candidate in candidates:
        value = _normalize_base_url(config.get(candidate))
        if value:
            return value

    for section_name in ("model_endpoints", "endpoints"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for candidate in candidates[2:]:
            value = _normalize_base_url(section.get(candidate))
            if value:
                return value
    return None


def configure(
    *,
    api_base_url: Optional[str] = None,
    endpoint_gem_1_bulk: Optional[str] = None,
    endpoint_gem_1_sc: Optional[str] = None,
    endpoint_gem_2: Optional[str] = None,
    model_endpoints: Optional[dict[str, str]] = None,
    self_hosted: Optional[bool] = None,
    reset: bool = False,
) -> None:
    """Configure default endpoints for the current Python process.

    Values passed here have higher precedence than environment variables and
    config files. Passing ``reset=True`` clears previous process-level settings
    before applying the provided values.
    """
    if reset:
        _CONFIG["api_base_url"] = None
        _CONFIG["self_hosted"] = None
        _CONFIG["model_endpoints"] = {}

    if api_base_url is not None:
        _CONFIG["api_base_url"] = _normalize_base_url(api_base_url)
    if self_hosted is not None:
        _CONFIG["self_hosted"] = bool(self_hosted)

    endpoints = {
        "gem-1-bulk": endpoint_gem_1_bulk,
        "gem-1-sc": endpoint_gem_1_sc,
        "gem-2": endpoint_gem_2,
    }
    if model_endpoints:
        endpoints.update({_base_model_id(k): v for k, v in model_endpoints.items()})

    for model_id, endpoint in endpoints.items():
        normalized = _normalize_base_url(endpoint)
        if normalized:
            _CONFIG["model_endpoints"][_base_model_id(model_id)] = normalized


def resolve_base_url(
    api_base_url: Optional[str] = None, model_id: Optional[str] = None
) -> str:
    """Resolve the API base URL for a request.

    Precedence: explicit per-call ``api_base_url`` arg > ``configure(...)`` >
    env vars > config file > production default.
    """
    explicit_base_url = _normalize_base_url(api_base_url)
    if explicit_base_url and explicit_base_url != API_BASE_URL:
        return explicit_base_url

    if model_id:
        configured = _configured_endpoint(model_id)
        if configured:
            return configured

    configured_base_url = _normalize_base_url(_CONFIG.get("api_base_url"))
    if configured_base_url:
        return configured_base_url

    if model_id:
        env_endpoint = _env_endpoint(model_id)
        if env_endpoint:
            return env_endpoint

    env_base_url = _normalize_base_url(os.environ.get(API_BASE_URL_ENV))
    if env_base_url:
        return env_base_url

    config = _file_config()
    if model_id:
        config_endpoint = _file_endpoint(config, model_id)
        if config_endpoint:
            return config_endpoint

    config_base_url = _normalize_base_url(config.get("api_base_url"))
    if config_base_url:
        return config_base_url

    return API_BASE_URL


def self_hosted_enabled(explicit: Optional[bool] = None) -> bool:
    """Resolve self-hosted mode: explicit arg > configure > env > config file."""
    if explicit is not None:
        return explicit
    if _CONFIG["self_hosted"] is not None:
        return bool(_CONFIG["self_hosted"])
    if SELF_HOSTED_ENV in os.environ:
        return env_flag(SELF_HOSTED_ENV)
    config = _file_config()
    if "self_hosted" in config:
        return bool(config["self_hosted"])
    return env_flag(SELF_HOSTED_ENV)


class SynthesizeAPIError(Exception):
    """Base exception for Synthesize API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(SynthesizeAPIError):
    """Raised when API authentication fails (401/403)."""

    pass


class NotFoundError(SynthesizeAPIError):
    """Raised when a resource is not found (404)."""

    pass


class ValidationError(SynthesizeAPIError):
    """Raised when the API returns a validation error (400/422)."""

    pass


def api_request(
    method: str,
    endpoint: str,
    api_base_url: str = API_BASE_URL,
    json: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """
    Make an authenticated request to the Synthesize API.

    Parameters
    ----------
    method : str
        HTTP method (GET, POST, etc.)
    endpoint : str
        API endpoint path (e.g., "/api/models")
    api_base_url : str, optional
        Base URL for the API server. Defaults to the production host.
    json : dict, optional
        JSON body to send with the request.
    timeout : int, optional
        Request timeout in seconds. Defaults to 30.

    Returns
    -------
    Any
        Parsed JSON response from the API.

    Raises
    ------
    KeyError
        If no API token is configured.
    AuthenticationError
        If the token is invalid (401/403).
    NotFoundError
        If the resource doesn't exist (404).
    ValidationError
        If the request is invalid (400/422).
    SynthesizeAPIError
        For other HTTP errors.
    """
    if not has_synthesize_token():
        raise KeyError(
            "No API token found. Set the SYNTHESIZE_API_KEY environment variable or "
            "call set_synthesize_token() before making API requests."
        )

    url = f"{api_base_url}{endpoint}"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {os.environ['SYNTHESIZE_API_KEY']}",
    }
    if json is not None:
        headers["Content-Type"] = "application/json"

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.HTTPError as err:
        status = err.response.status_code
        body = err.response.text

        if status in (401, 403):
            raise AuthenticationError(
                f"Authentication failed ({status}): {body}. "
                "Check that your SYNTHESIZE_API_KEY is valid.",
                status_code=status,
            ) from err
        elif status == 404:
            raise NotFoundError(
                f"Resource not found: {endpoint}",
                status_code=status,
            ) from err
        elif status in (400, 422):
            raise ValidationError(
                f"Invalid request ({status}): {body}",
                status_code=status,
            ) from err
        else:
            raise SynthesizeAPIError(
                f"API request failed ({status}): {body}",
                status_code=status,
            ) from err

    except requests.exceptions.RequestException as err:
        raise SynthesizeAPIError(f"Network error: {err}") from err


def _raise_self_hosted_http_error(err: "requests.exceptions.HTTPError", endpoint: str):
    """Map a self-hosted HTTP error to the appropriate SynthesizeAPIError."""
    status = err.response.status_code
    body = err.response.text
    if status in (401, 403):
        raise AuthenticationError(
            f"Authentication failed ({status}): {body}. "
            "The container has auth enabled; set SYNTHESIZE_API_KEY.",
            status_code=status,
        ) from err
    if status == 404:
        raise NotFoundError(
            f"Resource not found: {endpoint}", status_code=status
        ) from err
    if status in (400, 422):
        raise ValidationError(
            f"Invalid request ({status}): {body}", status_code=status
        ) from err
    raise SynthesizeAPIError(
        f"Self-hosted request failed ({status}): {body}", status_code=status
    ) from err


def open_arrow_stream(
    endpoint: str,
    api_base_url: str,
    json: dict,
    timeout: int = SELF_HOSTED_TIMEOUT,
) -> "requests.Response":
    """POST a query to a self-hosted container and return a streaming response.

    The response body is an Apache Arrow IPC stream that the caller reads
    incrementally from ``response.raw`` (``stream=True``), so the full payload
    is never buffered in memory at once -- this is what lets a partner generate
    more samples than would fit in RAM as a single buffer.

    Unlike :func:`api_request`, authentication is optional: a self-hosted
    container in an isolated network typically runs with auth disabled, so a
    token is attached only when ``SYNTHESIZE_API_KEY`` is set. The caller owns
    the returned response and must close it (use it as a context manager).
    """
    url = f"{api_base_url}{endpoint}"
    headers = {
        "Accept": ARROW_STREAM_CONTENT_TYPE,
        "Content-Type": "application/json",
    }
    token = os.environ.get("SYNTHESIZE_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.post(
            url, headers=headers, json=json, timeout=timeout, stream=True
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        _raise_self_hosted_http_error(err, endpoint)
    except requests.exceptions.RequestException as err:
        raise SynthesizeAPIError(f"Network error: {err}") from err

    # Decompress transparently if the server ever negotiates content-encoding.
    response.raw.decode_content = True
    return response


def get_self_hosted(
    endpoint: str,
    api_base_url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    """GET a JSON resource from a self-hosted container (auth optional).

    A token is attached only when ``SYNTHESIZE_API_KEY`` is set, so endpoints
    like ``/api/models`` and ``/api/models/{id}/example-query`` work against a
    no-auth container in an isolated network.
    """
    url = f"{api_base_url}{endpoint}"
    headers = {"Accept": "application/json"}
    token = os.environ.get("SYNTHESIZE_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        status = err.response.status_code
        body = err.response.text
        if status in (401, 403):
            raise AuthenticationError(
                f"Authentication failed ({status}): {body}. "
                "The container has auth enabled; set SYNTHESIZE_API_KEY.",
                status_code=status,
            ) from err
        if status == 404:
            raise NotFoundError(
                f"Resource not found: {endpoint}", status_code=status
            ) from err
        raise SynthesizeAPIError(
            f"Self-hosted request failed ({status}): {body}", status_code=status
        ) from err
    except requests.exceptions.RequestException as err:
        raise SynthesizeAPIError(f"Network error: {err}") from err


def get_json(url: str, timeout: int = DEFAULT_TIMEOUT) -> Any:
    """
    Fetch JSON from a URL (e.g., a signed download URL).

    This is for fetching from URLs that don't require authentication,
    such as pre-signed S3 URLs.

    Parameters
    ----------
    url : str
        The URL to fetch.
    timeout : int, optional
        Request timeout in seconds. Defaults to 30.

    Returns
    -------
    Any
        Parsed JSON response.

    Raises
    ------
    SynthesizeAPIError
        If the request fails or response is not valid JSON.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as err:
        raise SynthesizeAPIError(
            f"Download failed ({err.response.status_code}): {err.response.text}",
            status_code=err.response.status_code,
        ) from err
    except requests.exceptions.RequestException as err:
        raise SynthesizeAPIError(f"Network error: {err}") from err
    except ValueError as err:
        raise SynthesizeAPIError(
            f"Failed to decode JSON from response: {response.text}"
        ) from err
