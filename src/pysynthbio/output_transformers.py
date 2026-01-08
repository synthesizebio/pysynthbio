"""
Output transformers for different model types.

Each transformer is responsible for converting raw API JSON responses
into the appropriate output format for a specific model or model family.
"""

from typing import Dict, List, Tuple

import pandas as pd


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

    for key in ("outputs", "gene_order"):
        if key not in content:
            raise ValueError(f"Unexpected result JSON structure: {content}")

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


def transform_metadata_model_output(final_json: dict) -> List[Dict]:
    """
    Transformer for metadata prediction models.

    These models return classifier probabilities, latents, and metadata predictions
    rather than expression counts.

    Parameters
    ----------
    final_json : dict
        Raw JSON response from the API

    Returns
    -------
    dict
        List of MetadataOutput objects
    """

    return final_json["outputs"]


def transform_standard_model_output(final_json: dict) -> Dict:
    """
    Transformer for standard expression prediction models.

    Converts raw API JSON into structured DataFrames
    for expression, metadata, and latents.

    Parameters
    ----------
    final_json : dict
        Raw JSON response from the API

    Returns
    -------
    dict
        Dictionary containing metadata, expression, and latents DataFrames
    """
    expression, metadata, latents = _transform_result_to_frames(final_json)

    expression = expression.astype(int)

    # Build result dictionary - always include latents
    result = {"metadata": metadata, "expression": expression, "latents": latents}

    return result


# Registry mapping model_id to transformer function
OUTPUT_TRANSFORMERS = {
    "gem-1-bulk_predict-metadata": transform_metadata_model_output,
    "gem-1-sc_predict-metadata": transform_metadata_model_output,
    "gem-1-bulk": transform_standard_model_output,
    "gem-1-sc": transform_standard_model_output,
    "gem-1-bulk_reference-conditioning": transform_standard_model_output,
    "gem-1-sc_reference-conditioning": transform_standard_model_output,
}
