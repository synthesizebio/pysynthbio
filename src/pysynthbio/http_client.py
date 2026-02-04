"""Centralized HTTP client with consistent error handling for the Synthesize Bio API."""

import os
from typing import Any, Optional

import requests

from pysynthbio.key_handlers import has_synthesize_token

API_BASE_URL = "https://app.synthesize.bio"
DEFAULT_TIMEOUT = 30


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
