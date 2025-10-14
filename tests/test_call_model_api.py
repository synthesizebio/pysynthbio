import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from pysynthbio.call_model_api import (
        API_VERSION,
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
    Tests a live call to predict_query for the {API_VERSION} model.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    Requires the API server to be running at API_BASE_URL.
    NOTE: This is more of an integration test as it makes a real network call.
          For true unit tests, `requests.post` should be mocked.
    """
    print(f"\nTesting live predict_query call for {API_VERSION}...")

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
        print(f"predict_query call successful for {API_VERSION}.")
    except ValueError as e:
        pytest.fail(f"predict_query for {API_VERSION} raised ValueError: {e}")
    except KeyError as e:
        pytest.fail(
            f"predict_query for {API_VERSION} raised KeyError (API key issue?): {e}"
        )
    except Exception as e:
        pytest.fail(f"predict_query for {API_VERSION} raised unexpected Exception: {e}")

    assert isinstance(results, dict), f"Result for {API_VERSION} should be a dictionary"
    assert (
        "metadata" in results
    ), f"Result dictionary for {API_VERSION} should contain 'metadata' key"
    assert (
        "expression" in results
    ), f"Result dictionary for {API_VERSION} should contain 'expression' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]

    assert isinstance(
        metadata_df, pd.DataFrame
    ), f"'metadata' for {API_VERSION} should be a pandas DataFrame"
    assert isinstance(
        expression_df, pd.DataFrame
    ), f"'expression' for {API_VERSION} should be a pandas DataFrame"

    assert (
        not metadata_df.empty
    ), f"Metadata DataFrame for {API_VERSION} should not be empty for a valid query"
    assert (
        not expression_df.empty
    ), f"Expression DataFrame for {API_VERSION} should not be empty for a valid query"

    print(f"Assertions passed for {API_VERSION}.")


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_live_call_success_single_cell():
    """
    Tests a live call to predict_query for single-cell modality.
    Requires SYNTHESIZE_API_KEY to be set in the environment and the
    API server to support the single-cell async flow.
    """
    print(f"\nTesting live predict_query single-cell call for {API_VERSION}...")

    try:
        test_query = get_valid_query(modality="single-cell")
        print("Generated single-cell query:", test_query)
    except Exception as e:
        pytest.fail(f"get_valid_query(modality='single-cell') failed: {e}")

    try:
        results = predict_query(
            query=test_query,
            as_counts=True,
        )
        print("predict_query single-cell call successful.")
    except ValueError as e:
        pytest.fail(f"predict_query single-cell raised ValueError: {e}")
    except KeyError as e:
        pytest.fail(f"predict_query single-cell raised KeyError (API key issue?): {e}")
    except Exception as e:
        pytest.fail(f"predict_query single-cell raised unexpected Exception: {e}")

    assert isinstance(results, dict), "Single-cell result should be a dictionary"
    assert "metadata" in results, "Single-cell result should contain 'metadata' key"
    assert "expression" in results, "Single-cell result should contain 'expression' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]

    assert isinstance(metadata_df, pd.DataFrame), "'metadata' should be a DataFrame"
    assert isinstance(expression_df, pd.DataFrame), "'expression' should be a DataFrame"

    assert not metadata_df.empty, "Single-cell metadata DataFrame should not be empty"
    assert (
        not expression_df.empty
    ), "Single-cell expression DataFrame should not be empty"

    print("Assertions passed for single-cell live call.")


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_live_call_invalid_uberon():
    """
    Tests that the API properly rejects invalid UBERON IDs.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    """

    print(f"\nTesting live predict_query with invalid UBERON ID for {API_VERSION}...")

    # Create a query with an invalid UBERON ID
    invalid_query = {
        "inputs": [
            {
                "metadata": {
                    "tissue_ontology_id": "UBERON:9999999",  # Invalid ID
                    "age_years": "65",
                    "sex": "female",
                    "sample_type": "primary tissue",
                },
                "num_samples": 1,
            }
        ],
        "modality": "bulk",
        "mode": "sample generation",
    }

    # The API should reject this with a ValueError
    with pytest.raises(ValueError) as exc_info:
        predict_query(
            query=invalid_query,
            as_counts=True,
        )

    error_message = str(exc_info.value)
    print(f"API correctly rejected invalid UBERON ID with error: {error_message}")

    # The error message should now contain the validation details directly
    assert (
        "UBERON:9999999" in error_message
    ), f"Error message should mention the invalid UBERON ID. Got: {error_message}"
    assert (
        "bad values" in error_message.lower() or "invalid" in error_message.lower()
    ), f"Error message should indicate validation failure. Got: {error_message}"
    print("Successfully validated error message contains UBERON validation details")


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_live_call_invalid_uberon_single_cell():
    """
    Tests that the API properly rejects invalid UBERON IDs for single-cell modality.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    """

    print(
        (
            f"\nTesting predict_query (single-cell) "
            f"with invalid UBERON ID for {API_VERSION}..."
        )
    )

    # Create a single-cell query with an invalid UBERON ID
    invalid_query = {
        "inputs": [
            {
                "metadata": {
                    "cell_type_ontology_id": "CL:0000786",
                    "tissue_ontology_id": "UBERON:9999999",  # Invalid ID
                    "sex": "male",
                },
                "num_samples": 1,
            }
        ],
        "modality": "single-cell",
        "mode": "sample generation",
        "return_classifier_probs": True,
        "seed": 42,
    }

    # The API should reject this with a ValueError
    with pytest.raises(ValueError) as exc_info:
        predict_query(
            query=invalid_query,
            as_counts=True,
        )

    error_message = str(exc_info.value)
    print(
        (
            f"API correctly rejected invalid UBERON ID "
            f"(single-cell) with error: {error_message}"
        )
    )

    # The error message should now contain the validation details directly
    assert (
        "UBERON:9999999" in error_message
    ), f"Error message should mention the invalid UBERON ID. Got: {error_message}"
    assert (
        "bad values" in error_message.lower() or "invalid" in error_message.lower()
    ), f"Error message should indicate validation failure. Got: {error_message}"
    print(
        (
            "Successfully validated error message contains UBERON validation details "
            "(single-cell)"
        )
    )


# Add a mocked version of the API call test (bulk via async flow)
@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_mocked_call_success(mock_post, mock_get):
    """
    Tests a mocked call to predict_query for the {API_VERSION} model.
    This test doesn't require an API key or actual API server.
    """
    # Save the original API key state
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        # Ensure API key is set for this test
        os.environ["SYNTHESIZE_API_KEY"] = "mock-api-key-for-test"

        # POST /predict returns modelQueryId
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "bulk-xyz"}
        mock_post.return_value = post_resp

        # GETs: status running -> status ready -> download final JSON
        get_status_running = MagicMock()
        get_status_running.status_code = 200
        get_status_running.json.return_value = {"status": "running"}

        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/bulk.json",
        }

        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
            "outputs": [
                {
                    "counts": [100, 200, 300],
                    "metadata": {
                        "sample_id": "test1",
                        "cell_line_ontology_id": "CVCL_0023",
                        "age_years": "25",
                        "sex": "female",
                    },
                },
                {
                    "counts": [150, 250, 350],
                    "metadata": {
                        "sample_id": "test2",
                        "cell_line_ontology_id": "CVCL_0023",
                        "age_years": "30",
                        "sex": "male",
                    },
                },
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
            "model_version": 2,
        }
        mock_get.side_effect = [get_status_running, get_status_ready, get_download]

        print(f"\nTesting mocked predict_query call for {API_VERSION}...")

        try:
            test_query = get_valid_query()
            print("Generated query:", test_query)
        except Exception as e:
            pytest.fail(f"get_valid_query failed: {e}")

        try:
            results = predict_query(query=test_query, as_counts=True)
            print(f"predict_query mocked call successful for {API_VERSION}.")
        except Exception as e:
            pytest.fail(
                f"predict_query for {API_VERSION} raised unexpected Exception: {e}"
            )

        # Verify mocks were called
        mock_post.assert_called_once()

        assert isinstance(
            results, dict
        ), f"Result for {API_VERSION} should be a dictionary"
        assert (
            "metadata" in results
        ), f"Result dictionary for {API_VERSION} should contain 'metadata' key"
        assert (
            "expression" in results
        ), f"Result dictionary for {API_VERSION} should contain 'expression' key"

        metadata_df = results["metadata"]
        expression_df = results["expression"]

        assert isinstance(
            metadata_df, pd.DataFrame
        ), f"'metadata' for {API_VERSION} should be a pandas DataFrame"
        assert isinstance(
            expression_df, pd.DataFrame
        ), f"'expression' for {API_VERSION} should be a pandas DataFrame"

        # Check dimensions match new structure
        assert len(metadata_df) == 2, "Should have 2 metadata rows (one per output)"
        assert len(expression_df) == 2, "Should have 2 expression rows (one per output)"
        assert len(expression_df.columns) == 3, "Should have 3 gene columns"

        # Check data values
        assert list(expression_df.iloc[0]) == [100, 200, 300]
        assert list(expression_df.iloc[1]) == [150, 250, 350]

        print(f"Assertions passed for {API_VERSION} mocked call.")

    finally:
        # Restore original API key state
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]


# Add test for auto-authentication (bulk via async flow)
@patch("pysynthbio.call_model_api.set_synthesize_token")
@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_auto_authenticate(mock_post, mock_get, mock_set_token):
    """Test auto authentication in predict_query."""
    # Save the original API key state
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        # Ensure API key is not set initially
        if "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]

        # Configure mocks
        mock_set_token.side_effect = mock_set_token_implementation

        # POST /predict -> modelQueryId
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "bulk-abc"}
        mock_post.return_value = post_resp

        # Status ready -> then download
        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/final.json",
        }
        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
            "outputs": [
                {
                    "counts": [1, 2, 3],
                    "metadata": {"sample_id": "test1", "age_years": "25"},
                }
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
            "model_version": 2,
        }
        mock_get.side_effect = [get_status_ready, get_download]

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
    assert modalities == MODEL_MODALITIES[API_VERSION]


def test_get_valid_query_structure():
    """Tests get_valid_query returns the correct structure."""
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
    """Tests validate_query passes for a valid {API_VERSION} query."""
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


# Additional test to specifically verify the new data structure handling (bulk async)
@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_new_api_structure_handling(mock_post, mock_get):
    """Test that the updated code handles the new API structure correctly."""
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")

    try:
        os.environ["SYNTHESIZE_API_KEY"] = "mock-api-key-for-test"

        # POST returns modelQueryId
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "bulk-big"}
        mock_post.return_value = post_resp

        # Ready then huge download payload
        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/big.json",
        }
        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
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
        mock_get.side_effect = [get_status_ready, get_download]

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


# -----------------------------
# New tests for single-cell async flow
# -----------------------------


@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_single_cell_success(mock_post, mock_get):
    """Async single-cell happy path: running -> ready -> download JSON."""
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
    os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

    try:
        # POST /predict returns modelQueryId
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "abc123"}
        mock_post.return_value = post_resp

        # First GET status -> running; Second GET status -> ready with downloadUrl
        get_status_running = MagicMock()
        get_status_running.status_code = 200
        get_status_running.json.return_value = {"status": "running"}

        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/final.json",
        }

        # Third GET download -> final JSON with outputs + gene_order
        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
            "outputs": [
                {"counts": [1, 2, 3], "metadata": {"sample_id": "s1"}},
                {"counts": [4, 5, 6], "metadata": {"sample_id": "s2"}},
            ],
            "gene_order": ["gene1", "gene2", "gene3"],
        }

        mock_get.side_effect = [get_status_running, get_status_ready, get_download]

        q = get_valid_query(modality="single-cell")
        result = predict_query(q)

        assert "metadata" in result and "expression" in result
        assert len(result["metadata"]) == 2
        assert list(result["expression"].columns) == ["gene1", "gene2", "gene3"]
        assert list(result["expression"].iloc[0]) == [1, 2, 3]
        assert list(result["expression"].iloc[1]) == [4, 5, 6]
    finally:
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]


@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_single_cell_failure(mock_post, mock_get):
    """Async single-cell failure path: status -> failed with error message."""
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
    os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

    try:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "abc123"}
        mock_post.return_value = post_resp

        get_status_failed = MagicMock()
        get_status_failed.status_code = 200
        get_status_failed.json.return_value = {
            "status": "failed",
            "message": "Query validation failed: Invalid tissue_ontology_id",
        }
        mock_get.return_value = get_status_failed

        q = get_valid_query(modality="single-cell")
        with pytest.raises(ValueError, match="Model query failed"):
            predict_query(q)
    finally:
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]


@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_predict_query_single_cell_timeout(mock_post, mock_get):
    """Async single-cell timeout path: status remains running and timeout expires."""
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
    os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

    try:
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "abc123"}
        mock_post.return_value = post_resp

        # Always running; with timeout_seconds=0 we will exit immediately
        get_status_running = MagicMock()
        get_status_running.status_code = 200
        get_status_running.json.return_value = {"status": "running"}
        mock_get.return_value = get_status_running

        q = get_valid_query(modality="single-cell")
        with pytest.raises(ValueError, match="did not complete in time"):
            predict_query(
                q,
                poll_interval_seconds=0,
                poll_timeout_seconds=0,
            )
    finally:
        if original_api_key is not None:
            os.environ["SYNTHESIZE_API_KEY"] = original_api_key
        elif "SYNTHESIZE_API_KEY" in os.environ:
            del os.environ["SYNTHESIZE_API_KEY"]
