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
    assert (
        "latents" in results
    ), f"Result dictionary for {API_VERSION} should contain 'latents' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]
    latents_df = results["latents"]

    assert isinstance(
        metadata_df, pd.DataFrame
    ), f"'metadata' for {API_VERSION} should be a pandas DataFrame"
    assert isinstance(
        expression_df, pd.DataFrame
    ), f"'expression' for {API_VERSION} should be a pandas DataFrame"
    assert isinstance(
        latents_df, pd.DataFrame
    ), f"'latents' for {API_VERSION} should be a pandas DataFrame"

    assert (
        not metadata_df.empty
    ), f"Metadata DataFrame for {API_VERSION} should not be empty for a valid query"
    assert (
        not expression_df.empty
    ), f"Expression DataFrame for {API_VERSION} should not be empty for a valid query"
    assert (
        not latents_df.empty
    ), f"Latents DataFrame for {API_VERSION} should not be empty for a valid query"

    # Verify latents dimensions match the number of samples
    assert len(latents_df) == len(metadata_df), (
        f"Latents should have same number of rows as metadata "
        f"(got {len(latents_df)} vs {len(metadata_df)})"
    )

    # Verify latents contain actual data (not all zeros)
    # Latents is a DataFrame with columns like 'biological', 'technical', 'perturbation'
    # where each cell contains a list of floats
    assert len(latents_df.columns) > 0, "Latents should have at least one column"

    # Check that each latent type has non-zero values
    for col in latents_df.columns:
        # Each cell contains a list, so we need to flatten to check values
        all_values = []
        for cell_value in latents_df[col]:
            if isinstance(cell_value, list):
                all_values.extend(cell_value)

        values_sum = sum(all_values)
        latents_col_msg = (
            f"Latents column '{col}' should contain non-zero values, "
            f"but sum is {values_sum}"
        )
        assert values_sum != 0, latents_col_msg

        # Check for variation
        import numpy as np

        values_std = np.std(all_values)
        assert (
            values_std > 0
        ), f"Latents column '{col}' should have variation, but std is {values_std}"

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
    assert "latents" in results, "Single-cell result should contain 'latents' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]
    latents_df = results["latents"]

    assert isinstance(metadata_df, pd.DataFrame), "'metadata' should be a DataFrame"
    assert isinstance(expression_df, pd.DataFrame), "'expression' should be a DataFrame"
    assert isinstance(latents_df, pd.DataFrame), "'latents' should be a DataFrame"

    assert not metadata_df.empty, "Single-cell metadata DataFrame should not be empty"
    assert (
        not expression_df.empty
    ), "Single-cell expression DataFrame should not be empty"
    assert not latents_df.empty, "Single-cell latents DataFrame should not be empty"

    # Verify latents dimensions match the number of samples
    assert len(latents_df) == len(metadata_df), (
        f"Latents should have same number of rows as metadata "
        f"(got {len(latents_df)} vs {len(metadata_df)})"
    )

    # Verify latents contain actual data (not all zeros)
    # Latents is a DataFrame with columns like 'biological', 'technical', 'perturbation'
    # where each cell contains a list of floats
    assert len(latents_df.columns) > 0, "Latents should have at least one column"

    # Check that each latent type has non-zero values
    for col in latents_df.columns:
        # Each cell contains a list, so we need to flatten to check values
        all_values = []
        for cell_value in latents_df[col]:
            if isinstance(cell_value, list):
                all_values.extend(cell_value)

        values_sum = sum(all_values)
        latents_col_msg = (
            f"Latents column '{col}' should contain non-zero values, "
            f"but sum is {values_sum}"
        )
        assert values_sum != 0, latents_col_msg

        # Check for variation
        import numpy as np

        values_std = np.std(all_values)
        assert (
            values_std > 0
        ), f"Latents column '{col}' should have variation, but std is {values_std}"

    print("Assertions passed for single-cell live call.")


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_live_call_invalid_uberon_id():
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
        "mode": "mean estimation",  # Correct mode for single-cell
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
                    "latents": {
                        "biological": [0.5] * 1024,
                        "technical": [0.6] * 32,
                        "perturbation": [0.7] * 512,
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
                    "latents": {
                        "biological": [0.8] * 1024,
                        "technical": [0.9] * 32,
                        "perturbation": [1.0] * 512,
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
        assert (
            "latents" in results
        ), f"Result dictionary for {API_VERSION} should contain 'latents' key"

        metadata_df = results["metadata"]
        expression_df = results["expression"]
        latents_df = results["latents"]

        assert isinstance(
            metadata_df, pd.DataFrame
        ), f"'metadata' for {API_VERSION} should be a pandas DataFrame"
        assert isinstance(
            expression_df, pd.DataFrame
        ), f"'expression' for {API_VERSION} should be a pandas DataFrame"
        assert isinstance(
            latents_df, pd.DataFrame
        ), f"'latents' for {API_VERSION} should be a pandas DataFrame"

        # Check dimensions match new structure
        assert len(metadata_df) == 2, "Should have 2 metadata rows (one per output)"
        assert len(expression_df) == 2, "Should have 2 expression rows (one per output)"
        assert len(latents_df) == 2, "Should have 2 latents rows (one per output)"
        assert len(expression_df.columns) == 3, "Should have 3 gene columns"
        latents_types_msg = (
            "Should have 3 latent types (biological, technical, perturbation)"
        )
        assert len(latents_df.columns) == 3, latents_types_msg

        # Check data values
        assert list(expression_df.iloc[0]) == [100, 200, 300]
        assert list(expression_df.iloc[1]) == [150, 250, 350]

        # Latents now has dict structure with list-columns
        assert "biological" in latents_df.columns
        assert "technical" in latents_df.columns
        assert "perturbation" in latents_df.columns

        # Check that each cell contains a list
        assert isinstance(latents_df.iloc[0]["biological"], list)
        assert len(latents_df.iloc[0]["biological"]) == 1024
        assert len(latents_df.iloc[0]["technical"]) == 32
        assert len(latents_df.iloc[0]["perturbation"]) == 512

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
                    "latents": {
                        "biological": [0.1] * 1024,
                        "technical": [0.2] * 32,
                        "perturbation": [0.3] * 512,
                    },
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


def test_validate_query_single_cell_sample_generation():
    """Tests validate_query raises ValueError for single-cell with sample generation."""
    invalid_query = {
        "modality": "single-cell",
        "mode": "sample generation",  # Invalid for single-cell
        "inputs": [{"metadata": {}, "num_samples": 1}],
    }
    with pytest.raises(
        ValueError,
        match="Single-cell modality only supports 'mean estimation' mode",
    ):
        validate_query(invalid_query)
    print("validate_query correctly failed for single-cell with sample generation.")


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
                    "metadata": {
                        "age_years": "65",
                        "disease_ontology_id": "MONDO:0011719",
                        "sex": "female",
                        "sample_type": "primary tissue",
                        "tissue_ontology_id": "UBERON:0000945",
                    },
                    "latents": {
                        "biological": [1.0] * 1024,
                        "technical": [2.0] * 32,
                        "perturbation": [3.0] * 512,
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


@patch("pysynthbio.call_model_api.requests.get")
@patch("pysynthbio.call_model_api.requests.post")
def test_latents_extraction(mock_post, mock_get):
    """
    Test that latents are properly extracted from production API list format.
    """
    original_api_key = os.environ.get("SYNTHESIZE_API_KEY")
    os.environ["SYNTHESIZE_API_KEY"] = "test-api-token"

    try:
        # Mock the POST request to start the query
        post_resp = MagicMock()
        post_resp.status_code = 200
        post_resp.json.return_value = {"modelQueryId": "test-query-id"}
        mock_post.return_value = post_resp

        # Mock the GET requests for polling and download
        get_status_ready = MagicMock()
        get_status_ready.status_code = 200
        get_status_ready.json.return_value = {
            "status": "ready",
            "downloadUrl": "https://example.com/results.json",
        }

        # Mock response in production API format (list of dicts with latents)
        get_download = MagicMock()
        get_download.status_code = 200
        get_download.json.return_value = {
            "outputs": [
                {
                    "counts": [1, 2, 3, 4, 5],
                    "metadata": {
                        "sample_type": "cell line",
                        "cell_line_ontology_id": "CVCL_0023",
                    },
                    "latents": {
                        "biological": [0.1] * 1024,
                        "technical": [0.2] * 32,
                        "perturbation": [0.3] * 512,
                    },
                },
                {
                    "counts": [6, 7, 8, 9, 10],
                    "metadata": {
                        "sample_type": "primary tissue",
                        "tissue_ontology_id": "UBERON:0000945",
                    },
                    "latents": {
                        "biological": [0.6] * 1024,
                        "technical": [0.7] * 32,
                        "perturbation": [0.8] * 512,
                    },
                },
            ],
            "gene_order": [
                "ENSG00000000001",
                "ENSG00000000002",
                "ENSG00000000003",
                "ENSG00000000004",
                "ENSG00000000005",
            ],
            "model_version": 3,
        }
        mock_get.side_effect = [get_status_ready, get_download]

        # Test the predict_query function
        query = get_valid_query()
        results = predict_query(query, as_counts=True)

        # Verify latents are present and correctly extracted
        assert "latents" in results, "Results should contain 'latents' key"
        assert isinstance(
            results["latents"], pd.DataFrame
        ), "'latents' should be a pandas DataFrame"

        # Latents should have 2 rows (one per sample) and 3 columns
        # (biological, technical, perturbation)
        assert results["latents"].shape == (
            2,
            3,
        ), f"Expected latents shape (2, 3), got {results['latents'].shape}"

        # Check that the columns are the expected latent types
        expected_cols = {"biological", "technical", "perturbation"}
        assert (
            set(results["latents"].columns) == expected_cols
        ), f"Expected columns {expected_cols}, got {set(results['latents'].columns)}"

        # Verify metadata and expression are present and correct
        assert "metadata" in results
        assert "expression" in results
        assert results["metadata"].shape[0] == 2
        assert results["expression"].shape == (2, 5)

        print("Latents extraction test passed!")

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
                {
                    "counts": [1, 2, 3],
                    "metadata": {"sample_id": "s1"},
                    "latents": {
                        "biological": [0.1] * 1024,
                        "technical": [0.2] * 32,
                        "perturbation": [-0.1] * 512,
                    },
                },
                {
                    "counts": [4, 5, 6],
                    "metadata": {"sample_id": "s2"},
                    "latents": {
                        "biological": [0.3] * 1024,
                        "technical": [0.4] * 32,
                        "perturbation": [-0.2] * 512,
                    },
                },
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


# -----------------------------
# Biological validation tests with differential expression analysis
# -----------------------------


@pytest.mark.integration
@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_biological_validity_differential_expression_bulk():
    """
    Test that bulk RNA-seq data returns biologically valid expression data
    by performing simple differential expression analysis.
    """
    print("\nTesting biological validity with differential expression analysis...")

    # Import scipy for statistical tests
    try:
        from scipy import stats
    except ImportError:
        pytest.skip("scipy not available for statistical tests")

    # Create query with two distinct conditions
    de_query = {
        "inputs": [
            # Condition 1: Plasmacytoid dendritic cell
            {
                "metadata": {
                    # Plasmacytoid dendritic cell
                    "cell_type_ontology_id": "CL:0000784",
                    "tissue_ontology_id": "UBERON:0002371",  # bone marrow
                    "sex": "female",
                    "sample_type": "primary tissue",
                },
                "num_samples": 5,
            },
            # Condition 2: Myeloid cell
            {
                "metadata": {
                    "cell_type_ontology_id": "CL:0000763",  # Myeloid cell
                    "tissue_ontology_id": "UBERON:0002371",  # bone marrow
                    "sex": "female",
                    "sample_type": "primary tissue",
                },
                "num_samples": 5,
            },
        ],
        "modality": "bulk",
        "mode": "sample generation",
        "seed": 42,
    }

    results = predict_query(query=de_query, as_counts=True)

    # Split samples by condition
    group1_idx = list(range(5))
    group2_idx = list(range(5, 10))

    expr_group1 = results["expression"].iloc[group1_idx]
    expr_group2 = results["expression"].iloc[group2_idx]

    n_genes = results["expression"].shape[1]

    # Calculate mean expression for each group
    mean_group1 = expr_group1.mean(axis=0)
    mean_group2 = expr_group2.mean(axis=0)

    # Calculate fold changes (using pseudocount to avoid division by zero)
    pseudocount = 1
    fold_changes = np.log2((mean_group2 + pseudocount) / (mean_group1 + pseudocount))

    # Perform t-tests for each gene
    p_values = []
    for i in range(n_genes):
        try:
            _, p_val = stats.ttest_ind(expr_group1.iloc[:, i], expr_group2.iloc[:, i])
            p_values.append(p_val)
        except Exception:
            p_values.append(np.nan)

    p_values = np.array(p_values)

    # Basic validation of differential expression results
    print("Validating differential expression statistics...")

    # 1. Check that we have valid p-values
    valid_pvals = ~np.isnan(p_values)
    assert np.sum(valid_pvals) > n_genes * 0.9, (
        f"At least 90% of genes should have valid p-values "
        f"(got {np.sum(valid_pvals)}/{n_genes})"
    )

    # 2. P-values should be distributed between 0 and 1
    assert np.all(
        (p_values[valid_pvals] >= 0) & (p_values[valid_pvals] <= 1)
    ), "All p-values should be between 0 and 1"

    # 3. Not all p-values should be identical (showing variation)
    unique_pvals = len(np.unique(p_values[valid_pvals]))
    assert (
        unique_pvals > 100
    ), f"P-values should show variation (got {unique_pvals} unique values)"

    # 4. Fold changes should be reasonable (not all zero, not all extreme)
    fc_std = np.std(fold_changes[~np.isnan(fold_changes)])
    assert fc_std > 0, "Fold changes should show variation"

    fc_median = np.median(fold_changes[~np.isnan(fold_changes)])
    assert (
        abs(fc_median) < 10
    ), f"Median fold change should be reasonable (|log2FC| < 10, got {fc_median})"

    # 5. Check for differentially expressed genes (p < 0.05)
    de_genes = np.where(p_values < 0.05)[0]
    assert len(de_genes) > 0, "Should detect some differentially expressed genes"
    assert len(de_genes) < n_genes * 0.5, (
        f"Not all genes should be differentially expressed "
        f"(got {len(de_genes)}/{n_genes})"
    )

    # 6. Variance should exist within groups (biological variation)
    var_group1 = expr_group1.var(axis=0)
    var_group2 = expr_group2.var(axis=0)
    assert (
        np.median(var_group1[~np.isnan(var_group1)]) > 0
    ), "Group 1 should show within-group variance"
    assert (
        np.median(var_group2[~np.isnan(var_group2)]) > 0
    ), "Group 2 should show within-group variance"

    # 7. Expression levels should be reasonable for count data
    overall_mean = results["expression"].values.mean()
    assert overall_mean > 0, "Mean expression should be positive"
    assert (
        overall_mean < 1e6
    ), f"Mean expression should be in reasonable range (got {overall_mean})"

    print(
        f"DE analysis complete: {len(de_genes)} DE genes (p<0.05) "
        f"out of {np.sum(valid_pvals)} tested"
    )
    print(f"Median fold change: {fc_median:.3f} (log2)")
    print(
        f"Expression range: {results['expression'].values.min():.1f} "
        f"to {results['expression'].values.max():.1f}"
    )

    print("Biological validity tests passed!")


@pytest.mark.integration
@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_predict_query_biological_validity_differential_expression_single_cell():
    """
    Test that single-cell data returns biologically valid expression data
    by performing differential expression analysis.
    """
    print(
        "\nTesting single-cell biological validity with "
        "differential expression analysis..."
    )

    # Import scipy for statistical tests
    try:
        from scipy import stats
    except ImportError:
        pytest.skip("scipy not available for statistical tests")

    # Create query with two distinct cell types
    sc_de_query = {
        "inputs": [
            # Condition 1: T cells
            {
                "metadata": {
                    "cell_type_ontology_id": "CL:0000084",  # T cell
                    "tissue_ontology_id": "UBERON:0002371",  # bone marrow
                    "sex": "female",
                },
                "num_samples": 10,
            },
            # Condition 2: B cells
            {
                "metadata": {
                    "cell_type_ontology_id": "CL:0000236",  # B cell
                    "tissue_ontology_id": "UBERON:0002371",  # bone marrow
                    "sex": "female",
                },
                "num_samples": 10,
            },
        ],
        "modality": "single-cell",
        "mode": "mean estimation",
        "seed": 123,
    }

    results = predict_query(query=sc_de_query, as_counts=True)

    # Split samples by condition
    group1_idx = list(range(10))
    group2_idx = list(range(10, 20))

    expr_group1 = results["expression"].iloc[group1_idx]
    expr_group2 = results["expression"].iloc[group2_idx]

    n_genes = results["expression"].shape[1]
    n_cells = results["expression"].shape[0]

    print(f"Analyzing {n_cells} cells across {n_genes} genes...")

    # Single-cell specific metrics
    # 1. Calculate sparsity (proportion of zeros)
    sparsity_group1 = (expr_group1 == 0).values.sum() / (
        expr_group1.shape[0] * expr_group1.shape[1]
    )
    sparsity_group2 = (expr_group2 == 0).values.sum() / (
        expr_group2.shape[0] * expr_group2.shape[1]
    )

    assert sparsity_group1 > 0.3, (
        f"Single-cell data should show sparsity "
        f"(>30% zeros, got {sparsity_group1 * 100:.1f}%)"
    )
    assert sparsity_group1 < 0.95, (
        f"Single-cell data should not be too sparse "
        f"(<95% zeros, got {sparsity_group1 * 100:.1f}%)"
    )

    print(
        f"Sparsity: Group1 = {sparsity_group1 * 100:.1f}%, "
        f"Group2 = {sparsity_group2 * 100:.1f}%"
    )

    # 2. Calculate mean expression for each gene
    mean_group1 = expr_group1.mean(axis=0)
    mean_group2 = expr_group2.mean(axis=0)

    # 3. Calculate fold changes (using pseudocount for sparse data)
    pseudocount = 0.1
    fold_changes = np.log2((mean_group2 + pseudocount) / (mean_group1 + pseudocount))

    # 4. Perform Wilcoxon rank-sum tests (better for sparse/non-normal single-cell data)
    p_values = []
    for i in range(n_genes):
        try:
            _, p_val = stats.mannwhitneyu(
                expr_group1.iloc[:, i], expr_group2.iloc[:, i], alternative="two-sided"
            )
            p_values.append(p_val)
        except Exception:
            p_values.append(np.nan)

    p_values = np.array(p_values)

    # Validation of single-cell differential expression results
    print("Validating single-cell differential expression statistics...")

    # 1. Check that we have valid p-values
    valid_pvals = ~np.isnan(p_values)
    n_valid = np.sum(valid_pvals)
    assert n_valid > 100, f"Should have at least 100 testable genes (got {n_valid})"

    # 2. P-values should be distributed between 0 and 1
    assert np.all(
        (p_values[valid_pvals] >= 0) & (p_values[valid_pvals] <= 1)
    ), "All p-values should be between 0 and 1"

    # 3. P-values should show variation (not all the same)
    unique_pvals = len(np.unique(p_values[valid_pvals]))
    assert (
        unique_pvals > 100
    ), f"P-values should show variation (got {unique_pvals} unique values)"

    # 4. Fold changes should show variation
    fc_std = np.std(fold_changes[~np.isnan(fold_changes)])
    assert fc_std > 0, "Fold changes should show variation"

    fc_median = np.median(fold_changes[~np.isnan(fold_changes)])
    assert (
        abs(fc_median) < 15
    ), f"Median fold change should be reasonable for single-cell (got {fc_median})"

    # 5. Check for differentially expressed genes
    de_genes = np.where(p_values < 0.05)[0]
    assert len(de_genes) > 0, "Should detect some differentially expressed genes"
    assert len(de_genes) < n_genes * 0.6, (
        f"Not all genes should be differentially expressed "
        f"(got {len(de_genes)}/{n_genes})"
    )

    # 6. Check for genes with expression in at least some cells
    genes_expressed = (results["expression"] > 0).sum(axis=0)
    pct_expressed_genes = (genes_expressed > 0).mean() * 100
    assert pct_expressed_genes > 5, (
        f"At least 5% of genes should be expressed in some cells "
        f"(got {pct_expressed_genes:.1f}%)"
    )

    # 7. Single-cell specific: check for variance in expressed genes
    expressed_genes = (mean_group1 > 0) | (mean_group2 > 0)
    n_expressed = expressed_genes.sum()

    if n_expressed > 10:
        var_group1 = expr_group1.loc[:, expressed_genes].var(axis=0)
        mean_expressed = expr_group1.loc[:, expressed_genes].mean(axis=0)
        cv_group1 = np.sqrt(var_group1) / (mean_expressed + 1e-6)

        # For expressed genes, some should show variation
        high_cv = (cv_group1 > 0.1).sum()
        assert high_cv > 10, (
            f"Expressed genes should show variation "
            f"(got {high_cv} with CV>0.1 out of {n_expressed})"
        )

    # 8. Expression levels should be reasonable for single-cell count data
    overall_mean = results["expression"].values.mean()
    assert overall_mean > 0, "Mean expression should be positive"
    assert overall_mean < 1e5, (
        f"Mean expression should be in reasonable single-cell range "
        f"(got {overall_mean})"
    )

    # 9. Check that cell type markers might be differential
    strong_de = np.sum((np.abs(fold_changes) > 2) & (p_values < 0.01))
    assert (
        strong_de > 10
    ), f"Should detect some strongly DE genes between T and B cells (got {strong_de})"

    print(
        f"DE analysis complete: {len(de_genes)} DE genes (p<0.05) "
        f"out of {n_valid} tested"
    )
    print(f"Strongly DE genes (|log2FC|>2, p<0.01): {strong_de}")
    print(f"Median fold change: {fc_median:.3f} (log2)")
    print(
        f"Sparsity: {sparsity_group1 * 100:.1f}% (Group1), "
        f"{sparsity_group2 * 100:.1f}% (Group2)"
    )
    print(
        f"Expression range: {results['expression'].values.min():.1f} "
        f"to {results['expression'].values.max():.1f}"
    )

    print("Single-cell biological validity tests passed!")
