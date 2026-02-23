"""
Core API functionality for the Synthesize Bio API
"""

import time
from typing import Dict, Tuple

import pandas as pd

from pysynthbio.http_client import (
    API_BASE_URL,
    DEFAULT_TIMEOUT,
    SynthesizeAPIError,
    api_request,
    get_json,
)
from pysynthbio.key_handlers import has_synthesize_token, set_synthesize_token
from pysynthbio.output_transformers import OUTPUT_TRANSFORMERS

# Polling defaults for async model queries
DEFAULT_POLL_INTERVAL_SECONDS = 2
DEFAULT_POLL_TIMEOUT_SECONDS = 15 * 60


def _clean_error_message(message: str) -> str:
    """Strip server-side tracebacks from error messages."""
    if "\nTraceback" in message:
        return message.split("\nTraceback")[0].strip()
    return message


def predict_query(
    query: dict,
    model_id: str,
    auto_authenticate: bool = True,
    api_base_url: str = API_BASE_URL,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    return_download_url: bool = False,
    raw_response: bool = False,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Sends a query to the Synthesize Bio API for prediction and retrieves samples.

    Parameters
    ----------
    query : dict
        A dictionary representing the query data to send to the API.
        Use `get_valid_query()` to generate an example.

        The query dictionary supports the following optional parameters:

        - **total_count** (int): Library size used when converting predicted log CPM
          back to raw counts. Higher values scale counts up proportionally.
        - **deterministic_latents** (bool): If True, the model uses the mean of each
          latent distribution (p(z|metadata) or q(z|x)) instead of sampling.
          This removes randomness from latent sampling and produces deterministic
          outputs for the same inputs.

    model_id: str
        The model to use for prediction.

    auto_authenticate : bool, optional
        If True and no API token is found, will prompt the user to
        input one (default is True).
    api_base_url : str, optional
        Base URL for the API server. Defaults to the production host.
    poll_interval_seconds : int, optional
        Seconds between polling attempts of the status endpoint.
    poll_timeout_seconds : int, optional
        Maximum total seconds to wait before timing out.
    return_download_url : bool, optional
        If True, returns a dictionary with empty DataFrames without downloading
        or parsing the results. Default False.
    raw_response : bool, optional
        If True, returns the raw (unformatted) JSON response from the API
        without applying any output transformers. Default False.
    **kwargs : dict, optional
        Additional parameters to include in the query body. These are passed
        directly to the API and validated server-side.

    Returns
    -------
    dict
        metadata: pd.DataFrame (metadata, empty if return_download_url=True)
        expression: pd.DataFrame (expression, empty if return_download_url=True)
        latents: pd.DataFrame (latents from the model, empty if
            return_download_url=True)

    Raises
    ------
    KeyError
        If the SYNTHESIZE_API_KEY environment variable is not set and
        auto_authenticate is False.
    AuthenticationError
        If the API token is invalid.
    SynthesizeAPIError
        If API fails or response is invalid.
    """
    # Check if token is available and prompt if needed
    if not has_synthesize_token():
        if auto_authenticate:
            print("API token not found. Please provide your Synthesize Bio API token.")
            set_synthesize_token(use_keyring=True)
        else:
            raise KeyError(
                "No API token found. "
                "Set the SYNTHESIZE_API_KEY environment variable or "
                "call set_synthesize_token() before making API requests."
            )

    # Source field for reporting
    query["source"] = "pysynthbio"

    # Merge any additional kwargs into the query (validated server-side)
    query.update(kwargs)

    model_query_id = _start_model_query(
        api_base_url=api_base_url,
        model_id=model_id,
        query=query,
    )

    # Poll for completion
    status, payload = _poll_model_query(
        api_base_url=api_base_url,
        model_query_id=model_query_id,
        poll_interval=poll_interval_seconds,
        timeout_seconds=poll_timeout_seconds,
    )

    if status == "failed":
        # payload contains message with error details
        error_message = payload.get("message") if isinstance(payload, dict) else None
        if not error_message:
            raise SynthesizeAPIError("Model query failed. No error message in payload.")

        raise SynthesizeAPIError(
            f"Model query failed: {_clean_error_message(error_message)}"
        )

    if status != "ready":
        raise SynthesizeAPIError(
            (
                "Model query did not complete in time ("
                f"status={status}). Consider increasing "
                "poll_timeout_seconds."
            )
        )

    # When ready, payload should contain a signed downloadUrl to the final JSON
    download_url = payload.get("downloadUrl") if isinstance(payload, dict) else None
    if not download_url:
        raise SynthesizeAPIError("Response missing downloadUrl when status=ready")

    if return_download_url:
        return {
            "download_url": download_url,
        }

    # Fetch the final results JSON and transform to DataFrames
    final_json = get_json(download_url)

    if raw_response:
        return final_json

    transformer = OUTPUT_TRANSFORMERS.get(model_id)
    if transformer is None:
        raise ValueError(
            f"No output formatter registered for model_id '{model_id}'. "
            "To receive raw (unformatted) JSON, pass raw_response=True "
            "to predict_query()."
        )

    return transformer(final_json)


def _start_model_query(api_base_url: str, model_id: str, query: dict) -> str:
    """
    Starts an async model query and returns the modelQueryId.
    """
    content = api_request(
        method="POST",
        endpoint=f"/api/models/{model_id}/predict",
        api_base_url=api_base_url,
        json=query,
        timeout=DEFAULT_TIMEOUT,
    )

    if not isinstance(content, dict) or "modelQueryId" not in content:
        raise SynthesizeAPIError(
            f"Unexpected response from predict endpoint: {content}"
        )

    return str(content["modelQueryId"]).strip()


def _poll_model_query(
    api_base_url: str,
    model_query_id: str,
    poll_interval: int,
    timeout_seconds: int,
) -> Tuple[str, Dict[str, str]]:
    """
    Polls the status endpoint until ready/failed or timeout.

    Returns (status, payload) where payload may include downloadUrl or errorUrl.
    """
    start = time.time()
    endpoint = f"/api/model-queries/{model_query_id}/status"
    last_payload: Dict[str, str] = {}

    while True:
        payload = api_request(
            method="GET",
            endpoint=endpoint,
            api_base_url=api_base_url,
            timeout=DEFAULT_TIMEOUT,
        )

        if not isinstance(payload, dict) or "status" not in payload:
            raise SynthesizeAPIError(f"Unexpected status response: {payload}")

        status = str(payload.get("status"))
        last_payload = payload
        if status in ("ready", "failed"):
            return status, payload  # type: ignore[return-value]

        if (time.time() - start) > timeout_seconds:
            return status, last_payload

        time.sleep(max(1, int(poll_interval)))
