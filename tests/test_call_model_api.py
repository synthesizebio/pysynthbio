import os
import pytest
import pandas as pd
import numpy as np

try:
    from pysynthbio.call_model_api import (
        predict_query,
        get_valid_query,
        get_valid_modalities,
        validate_query,
        validate_modality,
        log_cpm,
        MODEL_MODALITIES,
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure the package is installed correctly (e.g., 'pip install -e .')")
    pytestmark = pytest.mark.skip(
        reason="Failed to import functions from call_model_api"
    )


api_key_available = "SYNTHESIZE_API_KEY" in os.environ
skip_reason_api_key = "SYNTHESIZE_API_KEY environment variable not set"


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_live_call_success():
    """
    Tests a live call to predict_query for the combined/v1.0 model.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    Requires the API server to be running at API_BASE_URL.
    NOTE: This is more of an integration test as it makes a real network call.
          For true unit tests, `requests.post` should be mocked.
    """
    print("\nTesting live predict_query call for combined v1.0...")

    try:
        test_query = get_valid_query()
        print("Generated query:", test_query)
    except Exception as e:
        pytest.fail(f"get_valid_query failed: {e}")

    try:
        results = predict_query(
            query=test_query,
            as_counts=True,
        )
        print("predict_query call successful for combined v1.0.")
    except ValueError as e:
        pytest.fail(f"predict_query for combined v1.0 raised ValueError: {e}")
    except KeyError as e:
        pytest.fail(
            f"predict_query for combined v1.0 raised KeyError (API key issue?): {e}"
        )
    except Exception as e:
        pytest.fail(f"predict_query for combined v1.0 raised unexpected Exception: {e}")

    assert isinstance(results, dict), "Result for combined v1.0 should be a dictionary"
    assert (
        "metadata" in results
    ), "Result dictionary for combined v1.0 should contain 'metadata' key"
    assert (
        "expression" in results
    ), "Result dictionary for combined v1.0 should contain 'expression' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]

    assert isinstance(
        metadata_df, pd.DataFrame
    ), "'metadata' for combined v1.0 should be a pandas DataFrame"
    assert isinstance(
        expression_df, pd.DataFrame
    ), "'expression' for combined v1.0 should be a pandas DataFrame"

    assert (
        not metadata_df.empty
    ), "Metadata DataFrame for combined v1.0 should not be empty for a valid query"
    assert (
        not expression_df.empty
    ), "Expression DataFrame for combined v1.0 should not be empty for a valid query"

    print("Assertions passed for combined v1.0.")


def test_get_valid_modalities():
    """Tests if get_valid_modalities returns the expected structure (a set)."""
    modalities = get_valid_modalities()
    assert isinstance(modalities, set)
    assert modalities == MODEL_MODALITIES["combined"]["v1.0"]


def test_get_valid_query_structure():
    """Tests get_valid_query returns the correct structure for the v1.0 model."""
    query = get_valid_query()
    expected_keys = {"inputs", "mode", "output_modality"}
    assert isinstance(query, dict)
    assert expected_keys.issubset(query.keys())
    assert isinstance(query["inputs"], list)


VALID_QUERY = {
    "inputs": [
        {
            "metadata": {
                "measurement": "measurement_1",
                "cell_line": "A549",
                "perturbation": "DMSO",
                "perturbation_type": "chemical",
                "perturbation_dose": "10 uM",
                "perturbation_time": "24 hours",
                "tissue": "lung",
                "cancer": True,
                "disease": "lung adenocarcinoma",
                "sex": "male",
                "age": "58 years",
                "ethnicity": "Caucasian",
                "sample_type": "cell line",
                "source": "ATCC",
            },
            "num_samples": 1,
        }
    ],
    "output_modality": "bulk_rna-seq",
    "mode": "mean estimation",
}


def test_validate_query_valid():
    """Tests validate_query passes for a valid v1.0 query."""
    try:
        validate_query(VALID_QUERY)
        print("validate_query passed as expected.")
    except (ValueError, TypeError) as e:
        pytest.fail(f"validate_query unexpectedly failed for valid query: {e}")


def test_validate_query_missing_keys():
    """Tests validate_query raises ValueError for missing keys."""
    invalid_query = VALID_QUERY.copy()
    del invalid_query["output_modality"]
    with pytest.raises(
        ValueError, match="Missing required keys in query: {'output_modality'}"
    ):
        validate_query(invalid_query)
    print("validate_query correctly failed for missing key.")


def test_validate_query_not_dict():
    """Tests validate_query raises TypeError if query is not a dict."""
    with pytest.raises(
        TypeError, match=r"Expected `query` to be a dictionary, but got \w+"
    ):
        validate_query("not a dict")
    print("validate_query correctly failed for non-dict query.")


def test_validate_modality_valid():
    """Tests validate_modality passes for a valid modality."""
    query = {"output_modality": "sra", "mode": "x", "inputs": []}
    try:
        validate_modality(query)
    except ValueError as e:
        pytest.fail(f"validate_modality raised ValueError unexpectedly: {e}")


def test_validate_modality_invalid():
    """Tests validate_modality raises ValueError for invalid modality."""
    query = {"output_modality": "invalid_modality", "mode": "x", "inputs": []}
    with pytest.raises(
        ValueError, match="Invalid modality 'invalid_modality'. Allowed modalities:"
    ):
        validate_modality(query)


def test_validate_modality_missing_key():
    """Tests validate_modality raises ValueError for missing modality key."""
    query = {"mode": "x", "inputs": []}
    with pytest.raises(ValueError, match="Query requires 'output_modality' key."):
        validate_modality(query)


def test_log_cpm():
    """Tests transforming counts to logCPM."""
    counts_data = pd.DataFrame(
        {"gene1": [1000000, 3000000], "gene2": [2000000, 6000000]}
    )
    expected_log_cpm = pd.DataFrame(
        {
            "gene1": [np.log1p(1e6 / 3), np.log1p(1e6 / 3)],
            "gene2": [np.log1p(2e6 / 3), np.log1p(2e6 / 3)],
        }
    )

    result_log_cpm = log_cpm(counts_data)
    pd.testing.assert_frame_equal(
        result_log_cpm, expected_log_cpm, check_dtype=False, rtol=1e-5
    )


def test_log_cpm_zero_counts():
    """Tests log_cpm handles rows with zero total counts."""
    counts_data = pd.DataFrame({"gene1": [10, 0], "gene2": [20, 0]})
    expected_log_cpm = pd.DataFrame(
        {
            "gene1": [np.log1p(10 / 30 * 1e6), 0.0],
            "gene2": [np.log1p(20 / 30 * 1e6), 0.0],
        }
    )
    result_log_cpm = log_cpm(counts_data)
    pd.testing.assert_frame_equal(
        result_log_cpm, expected_log_cpm, check_dtype=False, rtol=1e-5
    )
