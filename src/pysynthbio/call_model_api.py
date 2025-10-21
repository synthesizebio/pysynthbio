"""
Core API functionality for the Synthesize Bio API
"""

import json
import os
import time
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import requests

try:
    from .key_handlers import has_synthesize_token, set_synthesize_token
except ImportError:
    # Fallback if relative import fails (e.g., in tests)
    from pysynthbio.key_handlers import has_synthesize_token, set_synthesize_token

# Import package version and derive API version as v<major>.<minor>
try:
    from . import __version__ as _pkg_version
except Exception:
    _pkg_version = "0.0.0"

_API_VERSION_PARTS = _pkg_version.split(".")
API_VERSION = f"v{_API_VERSION_PARTS[0]}.{_API_VERSION_PARTS[1]}"

API_BASE_URL = "https://app.synthesize.bio"

MODEL_MODALITIES = {API_VERSION: {"bulk", "single-cell"}}

# Default timeout (seconds) for outbound HTTP requests
DEFAULT_TIMEOUT = 30

# Polling defaults for async model queries
DEFAULT_POLL_INTERVAL_SECONDS = 2
DEFAULT_POLL_TIMEOUT_SECONDS = 15 * 60


def get_valid_modalities() -> Set[str]:
    """
    Returns a set of possible output modalities for the supported model.

    Returns
    -------
    Set[str]
            A set containing the valid modality strings.
    """
    return MODEL_MODALITIES[API_VERSION]


def get_valid_modes() -> Set[str]:
    """
    Returns a set of possible output modes for the supported model.

    Returns
    -------
    Set[str]
            A set containing the valid modality strings.
    """
    return ["sample generation", "mean estimation", "metadata prediction"]


def get_valid_query(modality: str = "bulk") -> dict:
    """
    Generates a sample query for prediction and validation.

    Parameters
    ----------
    modality : str
        'bulk' or 'single-cell'. Defaults to 'bulk'.

    Returns
    -------
    dict
        A dictionary representing a valid query structure for the chosen
        modality.
    """
    if modality == "single-cell":
        return {
            "modality": "single-cell",
            "mode": "mean estimation",
            "seed": 11,
            "inputs": [
                {
                    "metadata": {
                        "cell_type_ontology_id": "CL:0000786",
                        "tissue_ontology_id": "UBERON:0001155",
                        "sex": "male",
                    },
                    "num_samples": 1,
                },
                {
                    "metadata": {
                        "cell_type_ontology_id": "CL:0000763",
                        "tissue_ontology_id": "UBERON:0001155",
                        "sex": "male",
                    },
                    "num_samples": 1,
                },
            ],
        }

    # Default: bulk
    return {
        "modality": "bulk",
        "mode": "sample generation",
        "seed": 11,
        "inputs": [
            {
                "metadata": {
                    "cell_line_ontology_id": "CVCL_0023",
                    "perturbation_ontology_id": "ENSG00000156127",
                    "perturbation_type": "crispr",
                    "perturbation_time": "96 hours",
                    "sample_type": "cell line",
                },
                "num_samples": 5,
            },
            {
                "metadata": {
                    "disease_ontology_id": "MONDO:0011719",
                    "age_years": "65",
                    "sex": "female",
                    "sample_type": "primary tissue",
                    "tissue_ontology_id": "UBERON:0000945",
                },
                "num_samples": 5,
            },
        ],
    }


def predict_query(
    query: dict,
    as_counts: bool = True,
    auto_authenticate: bool = True,
    api_base_url: str = API_BASE_URL,
    poll_interval_seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout_seconds: int = DEFAULT_POLL_TIMEOUT_SECONDS,
    return_download_url: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Sends a query to the Synthesize Bio API for prediction and retrieves samples.

    Parameters
    ----------
    query : dict
        A dictionary representing the query data to send to the API.
        Use `get_valid_query()` to generate an example.
    as_counts : bool, optional
        If False, transforms the predicted expression counts into
        logCPM (default is True, returning counts).
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

    Returns
    -------
    dict
        metadata: pd.DataFrame (metadata, empty if return_download_url=True)
        expression: pd.DataFrame (expression, empty if return_download_url=True)
        latents: pd.DataFrame (latents from the model, empty if
            return_download_url=True)

    Raises
    -------
    KeyError
        If the SYNTHESIZE_API_KEY environment variable is not set and
        auto_authenticate is False.
    ValueError
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
    # Validate base URL
    if not api_base_url.startswith("http"):
        raise ValueError(f"Invalid api_base_url: {api_base_url}")

    # Validate the query
    validate_query(query)

    # Ensure the query modality is valid
    validate_modality(query)
    modality = query["modality"]

    # Source field for reporting
    query["source"] = "pysynthbio"

    if modality in MODEL_MODALITIES[API_VERSION]:
        # Resolve internal API slug based on modality
        api_slug = _resolve_api_slug(modality)
        # Start async query
        model_query_id = _start_model_query(
            api_base_url=api_base_url,
            api_slug=api_slug,
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
            error_message = (
                payload.get("message") if isinstance(payload, dict) else None
            )
            if not error_message:
                raise ValueError("Model query failed. No error message in payload.")

            raise ValueError(f"Model query failed: {error_message}")

        if status != "ready":
            raise ValueError(
                (
                    "Model query did not complete in time ("
                    f"status={status}). Consider increasing "
                    "poll_timeout_seconds."
                )
            )

        # When ready, payload should contain a signed downloadUrl to the final JSON
        download_url = payload.get("downloadUrl") if isinstance(payload, dict) else None
        if not download_url:
            raise ValueError("Response missing downloadUrl when status=ready")

        if return_download_url:
            # Caller wants the URL only; return in a structured payload
            return {
                "metadata": pd.DataFrame(),
                "expression": pd.DataFrame(),
                "latents": pd.DataFrame(),
                "download_url": download_url,
            }

        # Fetch the final results JSON and transform to DataFrames
        final_json = _get_json(download_url)

        expression, metadata, latents = _transform_result_to_frames(final_json)

        expression = expression.astype(int)

        if not as_counts:
            expression = log_cpm(expression)

        # Build result dictionary - always include latents
        result = {"metadata": metadata, "expression": expression, "latents": latents}

        return result

    raise ValueError(
        (
            "Unsupported modality '"
            + str(modality)
            + "'. Expected one of "
            + str(MODEL_MODALITIES[API_VERSION])
        )
    )


def _resolve_api_slug(modality: str) -> str:
    if modality == "single-cell":
        return "gem-1-sc"
    if modality == "bulk":
        return "gem-1-bulk"
    raise ValueError(f"Unknown modality: '{modality}'")


def _start_model_query(api_base_url: str, api_slug: str, query: dict) -> str:
    """
    Starts an async model query and returns the modelQueryId.
    """
    try:
        response = requests.post(
            url=f"{api_base_url}/api/models/{api_slug}/predict",
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
                "Content-Type": "application/json",
            },
            json=query,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        content = response.json()
        if not isinstance(content, dict) or "modelQueryId" not in content:
            raise ValueError(f"Unexpected response from predict endpoint: {content}")
        return str(content["modelQueryId"]).strip()
    except requests.exceptions.HTTPError as err:
        raise ValueError(
            (
                f"Predict request failed with status "
                f"{err.response.status_code}: {err.response.text}"
            )
        ) from err
    except requests.exceptions.RequestException as err:
        raise ValueError(
            f"Predict request failed due to a network issue: {err}"
        ) from err


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
    status_url = f"{api_base_url}/api/model-queries/{model_query_id}/status"
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
    }
    last_payload: Dict[str, str] = {}
    while True:
        try:
            resp = requests.get(status_url, headers=headers, timeout=DEFAULT_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.HTTPError as err:
            raise ValueError(
                (
                    "Status request failed with status "
                    f"{err.response.status_code}: {err.response.text}"
                )
            ) from err
        except requests.exceptions.RequestException as err:
            raise ValueError(
                f"Status request failed due to a network issue: {err}"
            ) from err
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Failed to decode JSON from status response: {resp.text}"
            ) from err

        if not isinstance(payload, dict) or "status" not in payload:
            raise ValueError(f"Unexpected status response: {payload}")

        status = str(payload.get("status"))
        last_payload = payload
        if status in ("ready", "failed"):
            return status, payload  # type: ignore[return-value]

        if (time.time() - start) > timeout_seconds:
            return status, last_payload

        time.sleep(max(1, int(poll_interval)))


def _get_json(url: str) -> dict:
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as err:
        raise ValueError(
            (
                "Download URL fetch failed with status "
                f"{err.response.status_code}: {err.response.text}"
            )
        ) from err
    except requests.exceptions.RequestException as err:
        raise ValueError(
            f"Failed to fetch download URL due to a network issue: {err}"
        ) from err
    except json.JSONDecodeError as err:
        raise ValueError(
            (f"Failed to decode JSON from download URL response: {r.text}")
        ) from err


def _transform_result_to_frames(
    content: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transforms the final JSON result into (expression_df, metadata_df, latents_df).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - expression_df (pd.DataFrame): DataFrame of expression counts.
            - metadata_df (pd.DataFrame): DataFrame of sample metadata.
            - latents_df (pd.DataFrame): DataFrame of latents, which may be empty
              if not present in the response.
    """
    for key in ("error", "errors"):
        if key in content:
            raise ValueError(f"Error in result payload: {content[key]}")

    if "outputs" in content and "gene_order" in content:
        gene_order = content["gene_order"]
        outputs = content["outputs"]

        # Outputs is a list of dicts, each with
        # "counts" and "metadata"
        if not isinstance(outputs, list):
            raise ValueError(
                f"Unexpected outputs format: expected list, got {type(outputs)}. "
                "Please check API response structure."
            )

        expression_rows = []
        metadata_rows = []
        latents_rows = []

        for output in outputs:
            counts = output.get("counts", [])

            # Handle different response formats for counts
            if isinstance(counts, dict) and "counts" in counts:
                counts_list = counts["counts"]
            elif isinstance(counts, list):
                counts_list = counts
            else:
                # Single-cell format: dict mapping gene IDs to count values
                counts_list = [counts.get(gene, 0) for gene in gene_order]

            expression_rows.append(counts_list)
            metadata_rows.append(output.get("metadata", {}))

            # Extract latents if present in this output
            if "latents" in output:
                latents_rows.append(output["latents"])

        expression = pd.DataFrame(expression_rows, columns=gene_order)
        metadata = pd.DataFrame(metadata_rows)

        # Build latents DataFrame if any latents were found
        # Latents is a dict with keys like 'biological', 'technical', 'perturbation'
        # Each value is a list of floats. We create a DataFrame with these as columns
        # where each cell contains the list (similar to R's list-columns)
        if latents_rows:
            latents = pd.DataFrame(latents_rows)
        else:
            latents = pd.DataFrame()

        return expression.astype(int), metadata, latents

    raise ValueError(
        (
            "Unexpected result JSON structure (expected 'outputs' and 'gene_order'): "
            f"{content}"
        )
    )


def validate_query(query: dict) -> None:
    """
    Validates the structure and contents of the query based on the current API model.

    Parameters
    ----------
    query : dict
        The query dictionary.

    Raises
    -------
    TypeError
        If the query is not a dictionary.
    ValueError
        If the query is missing required keys for the current API model.
    """
    if not isinstance(query, dict):
        raise TypeError(
            f"Expected `query` to be a dictionary, but got {type(query).__name__}"
        )

    required_keys = {"inputs", "mode", "modality"}

    missing_keys = required_keys - query.keys()
    if missing_keys:
        raise ValueError(
            f"Missing required keys in query: {missing_keys}. "
            f"Use `get_valid_query()` to get an example."
        )

    # Validate single-cell modality only supports "mean estimation"
    if (
        query.get("modality") == "single-cell"
        and query.get("mode") == "sample generation"
    ):
        raise ValueError(
            "Single-cell modality only supports 'mean estimation' mode, "
            "not 'sample generation'. "
            "Please set mode='mean estimation' in your query."
        )


def validate_modality(query: dict) -> None:
    """
    Validates the modality in the query is allowed for the current API model.

    Parameters
    ----------
    query : dict
        A dictionary containing the query data.

    Raises
    -------
    ValueError
        If the modality key is missing, or the selected modality is not allowed.
    """
    allowed_modalities = MODEL_MODALITIES[API_VERSION]

    modality_key = "modality"
    if modality_key not in query:
        raise ValueError(f"Query requires '{modality_key}' key.")
    selected_modality = query[modality_key]

    if selected_modality not in allowed_modalities:
        raise ValueError(
            f"Invalid modality '{selected_modality}'. "
            f"Allowed modalities: {allowed_modalities}"
        )


def log_cpm(expression: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw counts expression data into log1p(CPM).

    Parameters
    ----------
    expression : pd.DataFrame
            A DataFrame containing raw counts expression data.

    Returns
    -------
    pd.DataFrame
            A DataFrame containing log1p(CPM) data.
    """
    expression_numeric = (
        expression.apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    )

    library_size = expression_numeric.sum(axis=1)

    non_zero_library = library_size > 0
    cpm = pd.DataFrame(0.0, index=expression.index, columns=expression.columns)

    if non_zero_library.any():
        cpm.loc[non_zero_library] = (
            expression_numeric.loc[non_zero_library].div(
                library_size[non_zero_library], axis=0
            )
            * 1e6
        )

    log_cpm_transformed = np.log1p(cpm)

    return log_cpm_transformed
