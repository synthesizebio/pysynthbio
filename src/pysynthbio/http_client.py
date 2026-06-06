"""Centralized HTTP client with consistent error handling for the Synthesize Bio API."""

import os
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


def env_flag(name: str) -> bool:
    """Interpret an env var as a boolean (1/true/yes/on, case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_base_url(api_base_url: Optional[str] = None) -> str:
    """Resolve the API base URL: explicit arg > env var > production default."""
    if api_base_url and api_base_url != API_BASE_URL:
        return api_base_url
    return os.environ.get(API_BASE_URL_ENV, API_BASE_URL)


def self_hosted_enabled(explicit: Optional[bool] = None) -> bool:
    """Resolve self-hosted mode: explicit arg wins, else the env flag."""
    if explicit is not None:
        return explicit
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
