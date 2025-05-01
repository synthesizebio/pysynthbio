import pandas as pd
import numpy as np
from functools import lru_cache
import os
import requests
import json
import re
import ast

PROXY_API_BASE_URL = "http://localhost"

MODEL_MODALITIES = {
    "rMetal": {
        "v0.2": {"bulk_rna-seq", "lincs", "sra"},
        "v0.4": {"bulk_rna-seq", "lincs", "sra"},
        "v0.5": {"bulk_rna-seq", "lincs", "sra"},
        "v0.6": {"bulk_rna-seq", "lincs", "sra"},
    },
    "DoGMA": {
        "v0.1": {"bulk_rna-seq", "lincs", "sra"},
        "v0.2": {"bulk_rna-seq", "lincs", "sra"},
        "v0.3": {"bulk_rna-seq", "lincs", "sra"},
    },
    "combined": {
        "v1.0": {
            "bulk_rna-seq",
            "lincs",
            "sra",
            "single_cell_rna-seq",
            "microarray",
            "pseudo_bulk",
        }
    },
}

DEFAULT_MODEL_FAMILY = "combined"
DEFAULT_MODEL_VERSION = "v1.0"

LOG_CPM_MODEL_TUPLES = {
    ("rMetal", "v0.2"),
    ("rMetal", "v0.4"),
    ("rMetal", "v0.5"),
    ("DoGMA", "v0.1"),
    ("DoGMA", "v0.2"),
}


def get_valid_modalities() -> dict:
    """
    Returns a dictionary of possible modalities per model family and version.

    Returns
    -------
    dict
        A nested dictionary containing the modalities.
    """
    return MODEL_MODALITIES


def get_available_models() -> list[dict]:
    """
    Retrieves the list of available models and their details from the proxy API.

    Returns
    -------
    list[dict]
        A list of dictionaries, where each dictionary contains details
        about an available model (e.g., name, version, description).
        Example: [{'name': 'rMetal', 'version': 'v0.2', ...}, ...]

    Raises
    -------
    KeyError
        If the SYNTHESIZE_API_KEY environment variable is not set.
    ValueError
        If the API request fails or the response format is invalid (not a list of dicts).
    """
    if "SYNTHESIZE_API_KEY" not in os.environ:
        raise KeyError("Please set the SYNTHESIZE_API_KEY environment variable")

    api_url = f"{PROXY_API_BASE_URL}/api/model"

    try:
        response = requests.get(
            url=api_url,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
            },
        )

        if response.status_code != 200:
            raise ValueError(
                f"API request to {api_url} failed with status {response.status_code}: {response.text}"
            )

        content = response.json()

        # Expecting a list of dictionaries like:
        # [{'name': 'rMetal', 'version': 'v0.2', ...}, ...]
        if not isinstance(content, list) or not all(isinstance(item, dict) for item in content):
            raise ValueError(f"API response from {api_url} is not a list of dictionaries: {content}")

        # Optionally: Validate required keys within each dict if needed, but for now just return the list
        # for item in content:
        #     if 'name' not in item or 'version' not in item:
        #         raise ValueError(f"Invalid model entry in API response, missing 'name' or 'version': {item}")

        return content # Return the list of dictionaries directly

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error calling {api_url}: {e}")
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from API response: {response.text}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        # Let's refine the error message here slightly
        raise ValueError(f"An unexpected error occurred while processing the response from {api_url}: {e}")


def get_valid_query(model_family: str, model_version: str) -> dict:
    """
    Generates a sample query for prediction and validation based on model version.

    Parameters
    ----------
    model_family: str
        Family name of the model (e.g., 'rMetal', 'combined').
    model_version: str
        Version string of the model (e.g., 'v0.6', 'v1.0').
    Returns
    -------
    dict
        A dictionary representing a valid query structure.
    """
    try:
        # Extract float from version string like "v1.0"
        # Ensure the version string starts with 'v' before slicing and converting
        if not model_version.startswith('v'):
            raise ValueError(f"Invalid model version format '{model_version}'. Expected prefix 'v'.")
        version_number = float(model_version[1:])
    except (ValueError, IndexError, TypeError) as e:
        # Catch broader errors including float conversion issues
        raise ValueError(f"Invalid model version format '{model_version}'. Failed to parse float: {e}")

    if version_number < 1.0:
        # Corresponds to RequestBodyV0.x schema
        return {
            "modality": "bulk_rna-seq",
            "num_samples": 5,
            "inputs": [
                str(
                    {
                        "cell_line": "A-549",
                        "perturbation": "ABL1",
                        "perturbation_type": "crispr",
                        "perturbation_time": "96 hours",
                        "sample_type": "cell line",
                    }
                ),
                str(
                    {
                        "disease": "gastrointestinal stromal tumor",
                        "age": "65 years",
                        "sex": "female",
                        "sample_type": "primary tissue",
                        "tissue": "stomach",
                    }
                ),
            ],
        }
    else:
        # Corresponds to RequestBodyV1.0 schema
        return {
            "output_modality": "sra",
            "mode": "mean estimation",
            "return_classifier_probs": True,
            "seed": 11,
            "inputs": [
                {
                    "metadata": {
                        "cell_line": "A-549",
                        "perturbation": "ABL1",
                        "perturbation_type": "crispr",
                        "perturbation_time": "96 hours",
                        "sample_type": "cell line",
                    },
                    "num_samples": 5,
                },
                {
                    "metadata": {
                        "disease": "gastrointestinal stromal tumor",
                        "age": "65 years",
                        "sex": "female",
                        "sample_type": "primary tissue",
                        "tissue": "stomach",
                    },
                    "num_samples": 5,
                },
            ],
        }


def predict_query(
    query: dict,
    model_family: str = DEFAULT_MODEL_FAMILY,
    model_version: str = DEFAULT_MODEL_VERSION,
    as_counts: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Sends a query to the Synthesize Bio Proxy API for prediction and retrieves samples.

    Parameters
    ----------
    query : dict
        A dictionary representing the query data to send to the API.
        Use `get_valid_query(family, version)` to generate an example.
    model_family : str, optional
        The family name of the model (e.g., 'rMetal', 'combined').
        Defaults to 'combined'.
    model_version : str, optional
        The version string of the model (e.g., 'v0.6', 'v1.0').
        Defaults to 'v1.0'.
    as_counts : bool, optional
        If True, transforms the predicted expression data into counts if the model
        returns logCPM (default is True). If the model returns counts and as_counts
        is False, transforms to logCPM.

    Returns
    -------
    dict
        metadata: pd.DataFrame containing metadata for each sample
        expression: pd.DataFrame containing expression data for each sample

    Raises
    -------
    KeyError
        If the SYNTHESIZE_API_KEY environment variable is not set.
    ValueError
        If API fails, model family/version is invalid, or response is invalid.
    """

    if "SYNTHESIZE_API_KEY" not in os.environ:
        raise KeyError("Please set the SYNTHESIZE_API_KEY environment variable")

    # Construct API URL using family and version
    if model_family and model_version:
        # Use the general template for all valid cases
        api_url = f"{PROXY_API_BASE_URL}/api/model/{model_family}/{model_version}"
        if model_family not in MODEL_MODALITIES or model_version not in MODEL_MODALITIES.get(model_family, {}):
            print(f"Warning: Model {model_family}/{model_version} not found in known MODEL_MODALITIES.")
    else:
        raise ValueError("Invalid model_family or model_version provided.")

    # Validate query structure and modality against the specific model
    validate_query(query, model_family, model_version)
    validate_modality(query, model_family, model_version)

    response = requests.post(
        url=api_url,
        headers={
            "Accept": "application/json",
            "Authorization": "Bearer " + os.environ["SYNTHESIZE_API_KEY"],
            "Content-Type": "application/json",
        },
        json=query,
    )

    # Check the HTTP status code
    if response.status_code != 200:
        raise ValueError(
            f"API request to {api_url} failed with status {response.status_code}: {response.text}"
        )
    # Parse the response JSON
    try:
        content = response.json()
        # Handle potential list wrapper for older versions if proxy preserves it
        if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
            content = content[0]
        elif not isinstance(content, dict):
             raise ValueError(f"API response is not a JSON object: {response.text}")

    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from API response: {response.text}")

    # Check for errors in the response body (common pattern)
    for key in ("error", "errors"):
        if key in content:
            raise ValueError(f"Error in response from API received: {content[key]}")

    # Determine response structure based on keys (adapt based on actual proxy response)
    if "outputs" in content and "gene_order" in content: # v1.0 structure
        expression = pd.concat(
            [
                pd.DataFrame(output["expression"], columns=content["gene_order"])
                for output in content["outputs"]
            ],
            ignore_index=True,
        )
        metadata_rows = [
            output["metadata"]
            for output in content["outputs"]
            for _ in range(len(output["expression"])) # Replicate metadata for each sample within the output
        ]
        metadata = pd.DataFrame(metadata_rows)

    elif "samples" in content: # Pre-v1.0 structure (assuming proxy might return this)
        samples = content["samples"]
        # Need modality to get gene order for pre-v1
        if "modality" not in query:
             raise ValueError("Query for pre-v1.0 model must contain 'modality' key.")
        modality = query["modality"]
        metadata = expand_metadata(query) # Uses the pre-v1.0 input structure
        expression = process_samples(samples, modality)
    else:
        raise ValueError(f"Unexpected API response structure: {content}")


    # --- Count transformation logic ---
    # Determine if the *queried model* typically returns logCPM
    # Check using tuple against the updated set
    model_returns_logcpm = (model_family, model_version) in LOG_CPM_MODEL_TUPLES

    if as_counts:
        if model_returns_logcpm:
            # Convert logCPM from model to counts
            expression = transform_to_counts(expression)
        else:
            # Model returns counts, user wants counts: ensure integer type
            expression = expression.astype(int)
    else: # User wants logCPM
        if model_returns_logcpm:
            # Model returns logCPM, user wants logCPM: do nothing (or ensure float type)
            expression = expression.astype(float) # Ensure float
        else:
            # Model returns counts, user wants logCPM: convert counts to logCPM
            expression = log_cpm(expression)

    return {"metadata": metadata, "expression": expression}


def validate_query(query: dict, model_family: str, model_version: str) -> None:
    """
    Validates the structure and contents of the query based on the target model version.

    Parameters
    ----------
    query : dict
        The query dictionary.
    model_family : str
        Family name of the model (e.g., 'rMetal', 'combined').
    model_version : str
        Version string of the model (e.g., 'v0.6', 'v1.0').

    Raises
    -------
    TypeError
        If the query is not a dictionary.
    ValueError
        If the query is missing required keys for the specific model version.
    """
    if not isinstance(query, dict):
        raise TypeError(
            f"Expected `query` to be a dictionary, but got {type(query).__name__}"
        )

    # Determine required keys based on model version
    try:
        version_number = float(model_version[1:]) # Extract float from e.g., "v1.0"
    except (ValueError, IndexError):
        # Cannot validate query structure if model version is invalid
        print(f"Warning: Skipping query validation due to invalid model version '{model_version}'.")
        return

    is_v1_or_later = version_number >= 1.0

    if is_v1_or_later:
        required_keys = {"inputs", "mode", "output_modality"}
    else:
        required_keys = {"inputs", "modality", "num_samples"}

    missing_keys = required_keys - query.keys()
    if missing_keys:
        # Construct combined name for error message
        model_name_combined = f"{model_family}{model_version}" if model_family != "combined" else "combinedv1.0"
        raise ValueError(
            f"Missing required keys in query for model '{model_name_combined}': {missing_keys}. "
            f"Use `get_valid_query('{model_family}', '{model_version}')` to get an example."
        )

    # Further validation could be added here (e.g., type checks for values)


def process_samples(samples: list, modality: str) -> pd.DataFrame:
    """
    Processes the samples returned from the API (pre-v1.0 format) into a DataFrame.

    Parameters
    ----------
    samples : list of lists
        A list of lists containing the expression data for each sample.
    modality : str
        The modality of the data (e.g. "bulk_rna-seq", "lincs", "sra"). Used to get gene order.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the expression data for all samples with ensembl gene id in columns.
    """
    gene_order = get_gene_order(modality)
    expression = pd.DataFrame(
        data=samples,
        columns=gene_order,
    ).clip(lower=0) # Ensure non-negative values

    return expression


@lru_cache()
def get_gene_order(modality) -> list:
    """
    Reads the gene order for a given modality from a local JSON file.

    Parameters
    ----------
    modality : str
        The modality for which to read the gene order (e.g., 'sra', 'bulk_rna-seq').

    Returns
    -------
    list
        A list of gene ids in the order expected/returned by the models.

    Raises
    -------
    FileNotFoundError
        If the gene order file is not found.
    KeyError
        If the modality is not found in the gene order file.
    """
    # Construct path relative to this file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming ai_gene_order.json is in a 'utils' subdir *relative to this script's location*
    # If 'utils' is at the project root, adjust accordingly.
    # For now, assuming utils/ is beside call_model_api.py needs fixing if structure is different.
    # Let's assume utils is at the *project root* relative to where pytest runs from.
    # This might be fragile.
    # A better approach might be using package resources if utils is part of the package data.

    # --- Option 1: Assume utils/ is at project root --- #
    # gene_order_file = os.path.join("utils", "ai_gene_order.json")

    # --- Option 2: Calculate path relative to this script --- #
    # Assuming src/utils/ai_gene_order.json structure
    # This is usually more robust when the package is installed.
    # gene_order_file = os.path.join(current_dir, "..", "utils", "ai_gene_order.json") # Old path, assumed utils was at root
    # gene_order_file = os.path.join(current_dir, "utils", "ai_gene_order.json") # Previous path, assumed utils was inside src/
    # Correct path assuming file is directly inside src/ alongside this script
    gene_order_file = os.path.join(current_dir, "ai_gene_order.json")
    gene_order_file = os.path.normpath(gene_order_file)


    try:
        with open(gene_order_file, "r") as file:
            data = json.load(file)
        if modality not in data:
             raise KeyError(f"Modality '{modality}' not found in gene order file: {gene_order_file}")
        return data[modality]
    except FileNotFoundError:
         raise FileNotFoundError(f"Gene order file not found at: {gene_order_file}")
    except json.JSONDecodeError:
         raise ValueError(f"Could not decode JSON from gene order file: {gene_order_file}")


def expand_metadata(query: dict) -> pd.DataFrame:
    """
    Replicates metadata for each sample in a pre-v1.0 query format.

    Parameters
    ----------
    query : dict
        A dictionary containing the query data in pre-v1.0 format
        (expects 'inputs' as list of strings, 'num_samples').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the metadata for each sample as different rows.

    Raises
    -------
    ValueError
        If 'inputs' are not strings or cannot be parsed, or if 'num_samples' is missing.
    """
    if "inputs" not in query or "num_samples" not in query:
         raise ValueError("Pre-v1.0 query format requires 'inputs' and 'num_samples' keys for metadata expansion.")

    try:
        dicts = [ast.literal_eval(item) for item in query["inputs"]]
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Could not parse metadata strings in 'inputs': {e}")

    num_samples = query["num_samples"]
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"'num_samples' must be a positive integer, got: {num_samples}")

    metadata = pd.DataFrame(
        [item for item in dicts for _ in range(num_samples)]
    )
    return metadata


def validate_modality(query: dict, model_family: str, model_version: str) -> None:
    """
    Validates the modality in the query is within the allowed modalities for the specified model.

    Parameters
    ----------
    query : dict
        A dictionary containing the query data.
    model_family : str
        Family name of the model (e.g., 'rMetal', 'combined').
    model_version : str
        Version string of the model (e.g., 'v0.6', 'v1.0').

    Raises
    -------
    ValueError
        If the model family/version is invalid, the modality key is missing, or the selected
        modality is not allowed for the model.
    """

    if model_family not in MODEL_MODALITIES or model_version not in MODEL_MODALITIES.get(model_family, {}):
        # If the model isn't explicitly listed, we might not be able to validate modalities.
        # Depending on requirements, could warn or raise an error.
        print(f"Warning: Cannot validate modality for unknown model {model_family}/{model_version}. Known families: {list(MODEL_MODALITIES.keys())}")
        return

    allowed_modalities = MODEL_MODALITIES[model_family][model_version]

    # Determine which modality key to check based on model version
    try:
        version_number = float(model_version[1:])
    except (ValueError, IndexError):
         raise ValueError(f"Cannot validate modality for invalid model version '{model_version}'.")

    is_v1_or_later = version_number >= 1.0

    if is_v1_or_later:
        modality_key = "output_modality"
        if modality_key not in query:
             # Construct combined name for error message
             model_name_combined = f"{model_family}{model_version}" if model_family != "combined" else "combinedv1.0"
             raise ValueError(f"Query for model '{model_name_combined}' requires '{modality_key}' key.")
        selected_modality = query[modality_key]
    else:
        modality_key = "modality"
        if modality_key not in query:
             # Construct combined name for error message
             model_name_combined = f"{model_family}{model_version}"
             raise ValueError(f"Query for model '{model_name_combined}' requires '{modality_key}' key.")
        selected_modality = query[modality_key]


    if selected_modality not in allowed_modalities:
        # Construct combined name for error message
        model_name_combined = f"{model_family}{model_version}" if model_family != "combined" else "combinedv1.0"
        raise ValueError(
            f"Invalid modality '{selected_modality}' for model '{model_name_combined}'. "
            f"Allowed modalities: {allowed_modalities}"
        )


def transform_to_counts(expression: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms expression data from log1p(CPM) into counts (approx. 30M total counts per sample).

    Parameters
    ----------
    expression : pd.DataFrame
        A DataFrame containing log1p(CPM) expression data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing estimated integer counts data.
    """
    # Ensure input is numeric and handle potential NaNs/Infs gracefully if necessary
    expression_numeric = expression.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Perform the transformation
    counts = (np.expm1(expression_numeric) * 30).round().astype(int) # Round before casting to int

    # Preserve original index and columns
    counts.index = expression.index
    counts.columns = expression.columns
    return counts


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
    # Ensure input is numeric and non-negative
    expression_numeric = expression.apply(pd.to_numeric, errors='coerce').fillna(0).clip(lower=0)

    # Calculate library size (total counts per sample)
    library_size = expression_numeric.sum(axis=1)

    # Avoid division by zero for samples with zero total counts
    # Add a small epsilon or handle these cases specifically (e.g., return zeros)
    non_zero_library = library_size > 0
    cpm = pd.DataFrame(0.0, index=expression.index, columns=expression.columns) # Initialize with zeros

    if non_zero_library.any():
        cpm.loc[non_zero_library] = expression_numeric.loc[non_zero_library].div(library_size[non_zero_library], axis=0) * 1e6

    # Calculate log1p(CPM)
    log_cpm_transformed = np.log1p(cpm)

    return log_cpm_transformed
