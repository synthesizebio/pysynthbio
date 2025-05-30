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
    Tests a live call to predict_query for the v2.0 model.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    Requires the API server to be running at API_BASE_URL.
    NOTE: This is more of an integration test as it makes a real network call.
          For true unit tests, `requests.post` should be mocked.
    """
    print("\nTesting live predict_query call for v2.0...")

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
        print("predict_query call successful for v2.0.")
    except ValueError as e:
        pytest.fail(f"predict_query for v2.0 raised ValueError: {e}")
    except KeyError as e:
        pytest.fail(f"predict_query for v2.0 raised KeyError (API key issue?): {e}")
    except Exception as e:
        pytest.fail(f"predict_query for v2.0 raised unexpected Exception: {e}")

    assert isinstance(results, dict), "Result for v2.0 should be a dictionary"
    assert (
        "metadata" in results
    ), "Result dictionary for v2.0 should contain 'metadata' key"
    assert (
        "expression" in results
    ), "Result dictionary for v2.0 should contain 'expression' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]

    assert isinstance(
        metadata_df, pd.DataFrame
    ), "'metadata' for v2.0 should be a pandas DataFrame"
    assert isinstance(
        expression_df, pd.DataFrame
    ), "'expression' for v2.0 should be a pandas DataFrame"

    assert (
        not metadata_df.empty
    ), "Metadata DataFrame for v2.0 should not be empty for a valid query"
    assert (
        not expression_df.empty
    ), "Expression DataFrame for v2.0 should not be empty for a valid query"

    print("Assertions passed for v2.0.")


# Add a mocked version of the API call test
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_mocked_call_success(mock_post):
    """
    Tests a mocked call to predict_query for the v2.0 model.
    This test doesn't require an API key or actual API server.
    """
    # Save the original API key state
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        # Ensure API key is set for this test
        os.environ["SYNTHESIZE_API_KEY"] = "mock-api-key-for-test"

        # Create mock response matching the NEW data structure
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "counts": [100, 200, 300],  # Now 1D list instead of 2D
                    "metadata": {
                        "sample_id": "test1",
                        "cell_line_ontology_id": "CVCL_0023",
                        "age_years": "25",
                        "sex": "female",
                    },
                    "classifier_probs": {
                        "sex": {"female": 0.8, "male": 0.2},
                        "age_years": {"20-30": 0.7, "30-40": 0.3},
                    },
                    "latents": {
                        "biological": [0.1, 0.2, 0.3],
                        "technical": [0.4, 0.5],
                        "perturbation": [0.6],
                    },
                },
                {
                    "counts": [150, 250, 350],  # Second sample
                    "metadata": {
                        "sample_id": "test2",
                        "cell_line_ontology_id": "CVCL_0023",
                        "age_years": "30",
                        "sex": "male",
                    },
                    "classifier_probs": {
                        "sex": {"female": 0.3, "male": 0.7},
                        "age_years": {"20-30": 0.4, "30-40": 0.6},
                    },
                    "latents": {
                        "biological": [0.2, 0.3, 0.4],
                        "technical": [0.5, 0.6],
                        "perturbation": [0.7],
                    },
                },
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
            "model_version": 2,
        }
        mock_post.return_value = mock_response

        print("\nTesting mocked predict_query call for v2.0...")

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
            print("predict_query mocked call successful for v2.0.")
        except Exception as e:
            pytest.fail(f"predict_query for v2.0 raised unexpected Exception: {e}")

        # Verify mock was called
        mock_post.assert_called_once()

        assert isinstance(results, dict), "Result for v2.0 should be a dictionary"
        assert (
            "metadata" in results
        ), "Result dictionary for v2.0 should contain 'metadata' key"
        assert (
            "expression" in results
        ), "Result dictionary for v2.0 should contain 'expression' key"

        metadata_df = results["metadata"]
        expression_df = results["expression"]

        assert isinstance(
            metadata_df, pd.DataFrame
        ), "'metadata' for v2.0 should be a pandas DataFrame"
        assert isinstance(
            expression_df, pd.DataFrame
        ), "'expression' for v2.0 should be a pandas DataFrame"

        # Check dimensions match new structure
        assert len(metadata_df) == 2, "Should have 2 metadata rows (one per output)"
        assert len(expression_df) == 2, "Should have 2 expression rows (one per output)"
        assert len(expression_df.columns) == 3, "Should have 3 gene columns"

        # Check data values
        assert list(expression_df.iloc[0]) == [100, 200, 300]
        assert list(expression_df.iloc[1]) == [150, 250, 350]

        print("Assertions passed for v2.0 mocked call.")

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

        # Create mock response with NEW structure
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "counts": [1, 2, 3],  # 1D list
                    "metadata": {"sample_id": "test1", "age_years": "25"},
                    "classifier_probs": {"sex": {"female": 0.6, "male": 0.4}},
                    "latents": {
                        "biological": [0.1, 0.2],
                        "technical": [0.3],
                        "perturbation": [0.4],
                    },
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
            "model_version": 2,
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

        # Verify dimensions
        assert len(results["metadata"]) == 1, "Should have 1 metadata row"
        assert len(results["expression"]) == 1, "Should have 1 expression row"

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
    assert modalities == MODEL_MODALITIES["v2.0"]


def test_get_valid_query_structure():
    """Tests get_valid_query returns the correct structure for the v2.0 model."""
    query = get_valid_query()
    expected_keys = {"inputs", "mode", "modality"}
    assert isinstance(query, dict)
    assert expected_keys.issubset(query.keys())
    assert isinstance(query["inputs"], list)


# Updated VALID_QUERY to match new structure expectations
VALID_QUERY = {
    "inputs": [
        {
            "metadata": {
                "cell_line_ontology_id": "CVCL_0023",
                "perturbation_ontology_id": "ENSG00000156127",
                "perturbation_type": "crispr",
                "perturbation_time": "96 hours",
                "sample_type": "cell line",
            },
            "num_samples": 1,
        }
    ],
    "modality": "bulk",
    "mode": "sample generation",
}


def test_validate_query_valid():
    """Tests validate_query passes for a valid v2.0 query."""
    try:
        validate_query(VALID_QUERY)
        print("validate_query passed as expected.")
    except (ValueError, TypeError) as e:
        pytest.fail(f"validate_query unexpectedly failed for valid query: {e}")


def test_validate_query_missing_keys():
    """Tests validate_query raises ValueError for missing keys."""
    invalid_query = VALID_QUERY.copy()
    del invalid_query["modality"]
    with pytest.raises(
        ValueError, match="Missing required keys in query: {'modality'}"
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
    query = {"modality": "bulk", "mode": "x", "inputs": []}
    try:
        validate_modality(query)
    except ValueError as e:
        pytest.fail(f"validate_modality raised ValueError unexpectedly: {e}")


def test_validate_modality_invalid():
    """Tests validate_modality raises ValueError for invalid modality."""
    query = {"modality": "invalid_modality", "mode": "x", "inputs": []}
    with pytest.raises(
        ValueError, match="Invalid modality 'invalid_modality'. Allowed modalities:"
    ):
        validate_modality(query)


def test_validate_modality_missing_key():
    """Tests validate_modality raises ValueError for missing modality key."""
    query = {"mode": "x", "inputs": []}
    with pytest.raises(ValueError, match="Query requires 'modality' key."):
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


# Additional test to specifically verify the new data structure handling
@patch("pysynthbio.call_model_api.requests.post")
def test_new_api_structure_handling(mock_post):
    """Test that the updated code handles the new API structure correctly."""
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        os.environ["SYNTHESIZE_API_KEY"] = "mock-api-key-for-test"

        # Create mock response that exactly matches your real API structure
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "outputs": [
                {
                    "counts": [0.1, 0.2, 0.3, 0.4, 0.5] * 8918,
                    "classifier_probs": {
                        "sex": {"female": 0.7, "male": 0.3},
                        "age_years": {"60-70": 0.8, "70-80": 0.2},
                        "tissue_ontology_id": {"UBERON:0000945": 0.9},
                    },
                    "latents": {
                        "biological": [-0.89, 0.38, -0.08] * 100,
                        "technical": [0.55, -0.38, -0.92] * 100,
                        "perturbation": [1.84, 0.46, -1.39] * 100,
                    },
                    "metadata": {
                        "age_years": "65",
                        "disease_ontology_id": "MONDO:0011719",
                        "sex": "female",
                        "sample_type": "primary tissue",
                        "tissue_ontology_id": "UBERON:0000945",
                    },
                }
            ],
            "gene_order": [f"ENSG{i:011d}" for i in range(44590)],
            "model_version": 2,
        }
        mock_post.return_value = mock_response

        # Test the predict_query function
        query = get_valid_query()
        results = predict_query(query, as_counts=True)

        # Verify the structure
        assert "metadata" in results
        assert "expression" in results
        assert len(results["metadata"]) == 1  # One row per output
        assert len(results["expression"]) == 1  # One row per output
        assert len(results["expression"].columns) == 44590  # All genes as columns

        print("New API structure handling test passed!")

    finally:
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]
