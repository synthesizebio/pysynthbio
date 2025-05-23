import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from pysynthbio.call_model_api import (
        MODEL_MODALITIES,
        get_valid_modalities,
        get_valid_query,
        log_cpm,
        predict_query,
        validate_modality,
        validate_query,
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure the package is installed correctly (e.g., 'pip install -e .')")
    pytestmark = pytest.mark.skip(
        reason="Failed to import functions from call_model_api"
    )


# Add this function to set up mocked authentication for tests
def mock_set_token_implementation(use_keyring=False):
    """Mock implementation that sets the environment variable"""
    os.environ["SYNTHESIZE_API_KEY"] = "mock-token-for-testing"
    return True


# Test for both live API calls (if API key available) and mocked calls
api_key_available = "SYNTHESIZE_API_KEY" in os.environ
skip_reason_api_key = "SYNTHESIZE_API_KEY environment variable not set"


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_live_call_success():
    """
    Tests a live call to predict_query for the v1.0 model.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    Requires the API server to be running at API_BASE_URL.
    NOTE: This is more of an integration test as it makes a real network call.
          For true unit tests, `requests.post` should be mocked.
    """
    print("\nTesting live predict_query call for v1.0...")

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
        print("predict_query call successful for v1.0.")
    except ValueError as e:
        pytest.fail(f"predict_query for v1.0 raised ValueError: {e}")
    except KeyError as e:
        pytest.fail(f"predict_query for v1.0 raised KeyError (API key issue?): {e}")
    except Exception as e:
        pytest.fail(f"predict_query for v1.0 raised unexpected Exception: {e}")

    assert isinstance(results, dict), "Result for v1.0 should be a dictionary"
    assert (
        "metadata" in results
    ), "Result dictionary for v1.0 should contain 'metadata' key"
    assert (
        "expression" in results
    ), "Result dictionary for v1.0 should contain 'expression' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]

    assert isinstance(
        metadata_df, pd.DataFrame
    ), "'metadata' for v1.0 should be a pandas DataFrame"
    assert isinstance(
        expression_df, pd.DataFrame
    ), "'expression' for v1.0 should be a pandas DataFrame"

    assert (
        not metadata_df.empty
    ), "Metadata DataFrame for v1.0 should not be empty for a valid query"
    assert (
        not expression_df.empty
    ), "Expression DataFrame for v1.0 should not be empty for a valid query"

    print("Assertions passed for v1.0.")


# Add a mocked version of the API call test
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_mocked_call_success(mock_post):
    """
    Tests a mocked call to predict_query for the v1.0 model.
    This test doesn't require an API key or actual API server.
    """
    # Save the original API key state
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        # Ensure API key is set for this test
        os.environ["SYNTHESIZE_API_KEY"] = "mock-api-key-for-test"

        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "metadata": {"sample_id": "test1", "cell_line": "A-549"},
                    "expression": [[1, 2, 3], [4, 5, 6]],
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
        }
        mock_post.return_value = mock_response

        print("\nTesting mocked predict_query call for v1.0...")

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
            print("predict_query mocked call successful for v1.0.")
        except Exception as e:
            pytest.fail(f"predict_query for v1.0 raised unexpected Exception: {e}")

        # Verify mock was called
        mock_post.assert_called_once()

        assert isinstance(results, dict), "Result for v1.0 should be a dictionary"
        assert (
            "metadata" in results
        ), "Result dictionary for v1.0 should contain 'metadata' key"
        assert (
            "expression" in results
        ), "Result dictionary for v1.0 should contain 'expression' key"

        metadata_df = results["metadata"]
        expression_df = results["expression"]

        assert isinstance(
            metadata_df, pd.DataFrame
        ), "'metadata' for v1.0 should be a pandas DataFrame"
        assert isinstance(
            expression_df, pd.DataFrame
        ), "'expression' for v1.0 should be a pandas DataFrame"

        print("Assertions passed for v1.0 mocked call.")

    finally:
        # Restore original API key state
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]


# Add test for auto-authentication
@patch("pysynthbio.call_model_api.set_synthesize_token")
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_auto_authenticate(mock_post, mock_set_token):
    """Test auto authentication in predict_query."""
    # Save the original API key state
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        # Ensure API key is not set initially
        if "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]

        # Configure mocks
        mock_set_token.side_effect = mock_set_token_implementation

        # Create mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "metadata": {"sample_id": "test1"},
                    "expression": [[1, 2, 3], [4, 5, 6]],
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
        }
        mock_post.return_value = mock_response

        print("\nTesting auto-authentication in predict_query...")

        # Get a valid query
        query = get_valid_query()

        # Call function with auto-authentication
        results = predict_query(query, auto_authenticate=True)

        # Verify set_token was called
        mock_set_token.assert_called_once_with(use_keyring=True)

        # Verify API was called
        mock_post.assert_called_once()

        # Verify results structure
        assert isinstance(results, dict)
        assert "metadata" in results
        assert "expression" in results

        print("Auto-authentication test passed.")

    finally:
        # Restore original API key state
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]


def test_get_valid_modalities():
    """Tests if get_valid_modalities returns the expected structure (a set)."""
    modalities = get_valid_modalities()
    assert isinstance(modalities, set)
    assert modalities == MODEL_MODALITIES["v1.0"]


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
