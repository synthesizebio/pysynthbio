"""
Core API functionality for the Synthesize Bio API
"""

import json
import os
from typing import Dict, Set

import numpy as np
import pandas as pd
import requests

try:
    from .key_handlers import has_synthesize_token, set_synthesize_token
except ImportError:
    # Fallback if relative import fails (e.g., in tests)
    from pysynthbio.key_handlers import has_synthesize_token, set_synthesize_token

API_BASE_URL = "https://app.synthesize.bio"

MODEL_MODALITIES = {"v2.0": {"bulk"}}


def get_valid_modalities() -> Set[str]:
    """
    Returns a set of possible output modalities for the supported model.

    Returns
    -------
    Set[str]
        A set containing the valid modality strings.
    """
    return MODEL_MODALITIES["v2.0"]


def get_valid_modes() -> Set[str]:
    """
    Returns a set of possible output modes for the supported model.

    Returns
    -------
    Set[str]
        A set containing the valid modality strings.
    """
    return ["sample generation", "mean estimation", "metadata prediction"]


def get_valid_query() -> dict:
    """
    Generates a sample query for prediction and validation for the v2.0 model.

    Returns
    -------
    dict
        A dictionary representing a valid query structure for v2.0.
    """
    return {
        "modality": "bulk",
        "mode": "sample generation",
        "return_classifier_probs": True,
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
    mode: str = "sample generation",
    as_counts: bool = True,
    auto_authenticate: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Sends a query to the Synthesize Bio API (v2.0) for
    prediction and retrieves samples.

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

    Returns
    -------
    dict
        metadata: pd.DataFrame containing metadata for each sample
        expression: pd.DataFrame containing expression data for each sample

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

    api_url = f"{API_BASE_URL}/api/model/v2.0"

    validate_query(query)

    response = requests.post(
        url=api_url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
            "Content-Type": "application/json",
        },
        json=query,
    )

    if response.status_code != 200:
        raise ValueError(
            f"API request to {api_url} failed with status",
            f"{response.status_code}: {response.text}",
        )
    try:
        content = response.json()
        if (
            isinstance(content, list)
            and len(content) == 1
            and isinstance(content[0], dict)
        ):
            content = content[0]
        elif not isinstance(content, dict):
            raise ValueError(f"API response is not a JSON object: {response.text}")

    except json.JSONDecodeError as err:
        raise ValueError(
            f"Failed to decode JSON from API response: {response.text}"
        ) from err

    for key in ("error", "errors"):
        if key in content:
            raise ValueError(f"Error in response from API received: {content[key]}")

    if "outputs" in content and "gene_order" in content:
        expression = pd.concat(
            [
                pd.DataFrame([output["counts"]], columns=content["gene_order"])
                # Wrap in [] to make it a single row
                for output in content["outputs"]
            ],
            ignore_index=True,
        )

        # Since each output now produces exactly 1 row, simplify metadata collection:
        metadata_rows = [output["metadata"] for output in content["outputs"]]
        metadata = pd.DataFrame(metadata_rows)
        metadata = pd.DataFrame(metadata_rows)
    else:
        raise ValueError(
            f"Unexpected API response structure "
            f"(expected 'outputs' and 'gene_order'): {content}"
        )

    expression = expression.astype(int)

    if not as_counts:
        expression = log_cpm(expression)

    return {"metadata": metadata, "expression": expression}


def validate_query(query: dict) -> None:
    """
    Validates the structure and contents of the query based on the v2.0 model.

    Parameters
    ----------
    query : dict
        The query dictionary.

    Raises
    -------
    TypeError
        If the query is not a dictionary.
    ValueError
        If the query is missing required keys for the v2.0 model.
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


def validate_modality(query: dict) -> None:
    """
    Validates the modality in the query is allowed for the v2.0 model.

    Parameters
    ----------
    query : dict
        A dictionary containing the query data.

    Raises
    -------
    ValueError
        If the modality key is missing, or the selected modality is not allowed.
    """
    allowed_modalities = MODEL_MODALITIES["v2.0"]

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
