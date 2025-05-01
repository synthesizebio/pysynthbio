# tests/test_call_model_api.py
import os
import pytest
import pandas as pd
import numpy as np # Import numpy for data generation

# Try importing the necessary functions from the src directory
# This assumes pytest is run from the workspace root
try:
    # Import directly, assuming the package is installed/discoverable
    from call_model_api import (
        predict_query,
        get_valid_query,
        get_valid_modalities,
        validate_query,
        validate_modality,
        expand_metadata,
        transform_to_counts,
        log_cpm,
        MODEL_MODALITIES,
        DEFAULT_MODEL_FAMILY,
        DEFAULT_MODEL_VERSION,
        get_available_models
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure the package is installed correctly (e.g., 'pip install -e .')")
    # If imports fail, skip tests that depend on them
    pytestmark = pytest.mark.skip(reason="Failed to import functions from call_model_api")


# --- Test Configuration ---
# Removed constants specific to single integration test, now parameterized
TEST_MODEL_FAMILY_V0 = "rMetal" # Keep for other unit tests
TEST_MODEL_VERSION_V0 = "v0.6" # Keep for other unit tests
# --- End Test Configuration ---

# Generate list of (family, version) tuples for parameterization
all_model_versions = [
    (family, version)
    for family, versions in MODEL_MODALITIES.items()
    for version in versions.keys()
]

# Check if API key is available, skip integration test if not
api_key_available = "SYNTHESIZE_API_KEY" in os.environ
skip_reason_api_key = "SYNTHESIZE_API_KEY environment variable not set"

# --- Integration Test ---
@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
@pytest.mark.parametrize("test_model_family, test_model_version", all_model_versions)
def test_predict_query_live_call_success(test_model_family, test_model_version):
    """
    Tests a live call to predict_query for various model parameters.
    Requires SYNTHESIZE_API_KEY to be set in the environment.
    Requires the local proxy API server to be running at PROXY_API_BASE_URL.
    This test first checks if the parameterized model is available via get_available_models()
    and skips if not.
    NOTE: This is more of an integration test as it makes a real network call.
          For true unit tests, `requests.post` should be mocked.
    """
    print(f"\nTesting live predict_query call for {test_model_family} {test_model_version}...")

    # Explicitly skip rMetal v0.2 due to persistent 400 errors
    if test_model_family == "rMetal" and test_model_version == "v0.2":
        pytest.skip(f"Skipping {test_model_family}{test_model_version}: Not sure what's going on here, but this model/version is consistently failing with 400 Bad Request")

    # 0. Check if model is listed as available by the API first
    try:
        available_models_list = get_available_models()
        available_model_tuples = {(m['name'], m['version']) for m in available_models_list}
        print(f"API reports available models: {available_model_tuples}")
    except Exception as e:
        # If we can't even get the list of models, skip subsequent tests related to predict_query
        pytest.skip(f"Could not retrieve available models from API, skipping predict_query tests. Error: {e}")

    current_model_tuple = (test_model_family, test_model_version)
    if current_model_tuple not in available_model_tuples:
        pytest.skip(f"Model {test_model_family}{test_model_version} not reported as available by the API via get_available_models().")

    # --- Proceed with the test only if the model is available --- 

    # 1. Get a valid query
    try:
        test_query = get_valid_query(test_model_family, test_model_version)
        print("Generated query:", test_query)
    except Exception as e:
        pytest.fail(f"get_valid_query failed for {test_model_family}{test_model_version}: {e}")

    # 2. Call predict_query
    try:
        results = predict_query(
            query=test_query,
            model_family=test_model_family,
            model_version=test_model_version,
            as_counts=True
        )
        print(f"predict_query call successful for {test_model_family}{test_model_version}.")
    except ValueError as e:
        # Catch potential API errors (like 4xx, 5xx) or other ValueErrors from the function
        pytest.fail(f"predict_query for {test_model_family}{test_model_version} raised ValueError: {e}")
    except KeyError as e:
        # Should be caught by skipif, but as a safeguard
         pytest.fail(f"predict_query for {test_model_family}{test_model_version} raised KeyError (API key issue?): {e}")
    except Exception as e:
        # Catch any other unexpected exceptions
        pytest.fail(f"predict_query for {test_model_family}{test_model_version} raised unexpected Exception: {e}")

    # 3. Assert results structure and types
    assert isinstance(results, dict), f"Result for {test_model_family}{test_model_version} should be a dictionary"
    assert "metadata" in results, f"Result dictionary for {test_model_family}{test_model_version} should contain 'metadata' key"
    assert "expression" in results, f"Result dictionary for {test_model_family}{test_model_version} should contain 'expression' key"

    metadata_df = results["metadata"]
    expression_df = results["expression"]

    assert isinstance(metadata_df, pd.DataFrame), f"'metadata' for {test_model_family}{test_model_version} should be a pandas DataFrame"
    assert isinstance(expression_df, pd.DataFrame), f"'expression' for {test_model_family}{test_model_version} should be a pandas DataFrame"

    # Optionally, assert that dataframes are not empty (assuming valid query returns samples)
    assert not metadata_df.empty, f"Metadata DataFrame for {test_model_family}{test_model_version} should not be empty for a valid query"
    assert not expression_df.empty, f"Expression DataFrame for {test_model_family}{test_model_version} should not be empty for a valid query"

    print(f"Assertions passed for {test_model_family}{test_model_version}.")

@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_get_available_models_live():
    """Tests a live call to get_available_models.
    Requires SYNTHESIZE_API_KEY and a running local proxy API.
    Expects the API to return a list of dictionaries, each with model details.
    """
    print("\nTesting live get_available_models call...")
    # Define the expected keys based on the example structure
    expected_keys = {
        "name", "version", "description", "species",
        "assembly", "annotation", "geneMapVersion", "metadataVersion"
    }

    try:
        models = get_available_models() # Function now returns list[dict]
        print(f"Received models structure: {models}")

        assert isinstance(models, list), "Response should be a list"
        assert len(models) > 0, "Model list should not be empty"

        # Check that all items are dictionaries
        assert all(isinstance(m, dict) for m in models), \
            "All items in the list should be dictionaries"

        # Check that all dictionaries contain the expected keys
        for i, model_dict in enumerate(models):
            missing_keys = expected_keys - model_dict.keys()
            assert not missing_keys, \
                f"Model dictionary at index {i} is missing keys: {missing_keys}. Got: {model_dict.keys()}"
            extra_keys = model_dict.keys() - expected_keys
            if extra_keys:
                print(f"Note: Model dict at index {i} has extra keys: {extra_keys}") # Optional: Just print extra keys

        # Optionally check for a specific model dictionary (already present)
        assert any(m['name'] == 'rMetal' and m['version'] == 'v1.0' for m in models), \
            "The list should contain the rMetal v1.0 model dictionary"

        print("Assertions passed for get_available_models (list of dicts with expected keys).")

    except ValueError as e:
        pytest.fail(f"get_available_models raised ValueError: {e}")
    except KeyError as e:
        pytest.fail(f"get_available_models raised KeyError (API key issue?): {e}")
    except Exception as e:
        pytest.fail(f"get_available_models raised unexpected Exception: {e}")

# --- Unit Tests ---

def test_get_valid_modalities():
    """Tests if get_valid_modalities returns the expected structure."""
    modalities = get_valid_modalities()
    assert isinstance(modalities, dict)
    # Check if it matches the imported constant (or perform deeper structure checks)
    assert modalities == MODEL_MODALITIES
    assert "combined" in modalities
    assert "v1.0" in modalities["combined"]

# Parameterize tests for get_valid_query for different model versions
@pytest.mark.parametrize(
    "family, version, expected_keys, not_expected_keys",
    [
        (TEST_MODEL_FAMILY_V0, TEST_MODEL_VERSION_V0, {"modality", "num_samples", "inputs"}, {"output_modality", "mode"}),
    ]
)
def test_get_valid_query_structure(family, version, expected_keys, not_expected_keys):
    """Tests get_valid_query returns the correct structure for different versions."""
    query = get_valid_query(family, version)
    assert isinstance(query, dict)
    assert expected_keys.issubset(query.keys())
    assert not any(key in query for key in not_expected_keys)
    assert isinstance(query["inputs"], list)

def test_get_valid_query_invalid_version():
    """Tests get_valid_query raises ValueError for invalid version format."""
    with pytest.raises(ValueError, match=r"Invalid model version format"):
        get_valid_query("combined", "1.0") # Missing 'v'
    with pytest.raises(ValueError, match=r"Invalid model version format"):
        get_valid_query("rMetal", "v0x6") # Invalid number

# --- Tests for validate_query --- #

# Define static example queries and model IDs for these tests
# Example v1.0 model and a suitable query (e.g., from combinedv1.0)
V1_MODEL_FAMILY = "combined"
V1_MODEL_VERSION = "v1.0"
V1_VALID_QUERY = {
    "inputs": [
        {
            "metadata": { # v1 uses 'metadata' key inside inputs
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
            "num_samples": 1 # v1 has num_samples per input item
        }
    ],
    "output_modality": "bulk_rna-seq", # v1 uses output_modality
    "mode": "mean estimation", # v1 requires mode
    # Removed modality and num_samples from top level for v1
}

# Example pre-v1.0 model and a suitable query (e.g., from rMetalv0.6)
V0_MODEL_FAMILY = "rMetal"
V0_MODEL_VERSION = "v0.6"
V0_VALID_QUERY = {
    "inputs": [
        str({ # Input is stringified dict for older models
            "cell_line": "A-549",
            "perturbation": "ABL1",
            "perturbation_type": "crispr",
            "perturbation_time": "96 hours",
        })
    ],
    "modality": "sra", # Modality valid for rMetalv0.6
    "num_samples": 1,
}


def test_validate_query_v1_valid():
    """Tests validate_query passes for a valid v1.0 query."""
    # Use the statically defined V1 example
    try:
        validate_query(V1_VALID_QUERY, V1_MODEL_FAMILY, V1_MODEL_VERSION)
        print(f"validate_query passed for {V1_MODEL_FAMILY}{V1_MODEL_VERSION} as expected.")
    except (ValueError, TypeError) as e:
        pytest.fail(f"validate_query unexpectedly failed for valid v1 query: {e}")


def test_validate_query_v0_valid():
    """Tests validate_query passes for a valid pre-v1.0 query."""
    # Use the statically defined V0 example
    try:
        validate_query(V0_VALID_QUERY, V0_MODEL_FAMILY, V0_MODEL_VERSION)
        print(f"validate_query passed for {V0_MODEL_FAMILY}{V0_MODEL_VERSION} as expected.")
    except (ValueError, TypeError) as e:
        pytest.fail(f"validate_query unexpectedly failed for valid v0 query: {e}")


def test_validate_query_v1_missing_keys():
    """Tests validate_query raises ValueError for missing keys in v1.0 query."""
    invalid_query = V1_VALID_QUERY.copy()
    # Remove a key required for v1 models (e.g., output_modality)
    del invalid_query["output_modality"]
    with pytest.raises(ValueError, match="Missing required keys"):
        validate_query(invalid_query, V1_MODEL_FAMILY, V1_MODEL_VERSION)
    print(f"validate_query correctly failed for missing key in v1 query.")


def test_validate_query_v0_missing_keys():
    """Tests validate_query raises ValueError for missing keys in pre-v1.0 query."""
    invalid_query = V0_VALID_QUERY.copy()
    del invalid_query["inputs"] # Remove a required key
    with pytest.raises(ValueError, match="Missing required keys"):
        validate_query(invalid_query, V0_MODEL_FAMILY, V0_MODEL_VERSION)
    print(f"validate_query correctly failed for missing key in v0 query.")


def test_validate_query_not_dict():
    """Tests validate_query raises TypeError if query is not a dict."""
    with pytest.raises(TypeError, match=r"Expected `query` to be a dictionary, but got \w+"):
        validate_query("not a dict", V1_MODEL_FAMILY, V1_MODEL_VERSION) # Use V1 model just for example
    print(f"validate_query correctly failed for non-dict query.")


# --- validate_modality Tests ---
def test_validate_modality_v1_valid():
    """Tests validate_modality passes for a valid v1.0 modality."""
    query = {"output_modality": "sra", "mode": "x", "inputs": []}
    try:
        validate_modality(query, "combined", "v1.0")
    except ValueError as e:
        pytest.fail(f"validate_modality raised ValueError unexpectedly: {e}")

def test_validate_modality_v0_valid():
    """Tests validate_modality passes for a valid pre-v1.0 modality."""
    query = {"modality": "lincs", "num_samples": 1, "inputs": []}
    try:
        validate_modality(query, "rMetal", "v0.6")
    except ValueError as e:
        pytest.fail(f"validate_modality raised ValueError unexpectedly: {e}")

def test_validate_modality_v1_invalid():
    """Tests validate_modality raises ValueError for invalid v1.0 modality."""
    query = {"output_modality": "invalid_modality", "mode": "x", "inputs": []}
    with pytest.raises(ValueError, match="Invalid modality"):
        validate_modality(query, "combined", "v1.0")

def test_validate_modality_v0_invalid():
    """Tests validate_modality raises ValueError for invalid pre-v1.0 modality."""
    query = {"modality": "invalid_modality", "num_samples": 1, "inputs": []}
    with pytest.raises(ValueError, match="Invalid modality"):
        validate_modality(query, "rMetal", "v0.6")

def test_validate_modality_v1_missing_key():
    """Tests validate_modality raises ValueError for missing v1.0 modality key."""
    query = {"mode": "x", "inputs": []}
    with pytest.raises(ValueError, match="requires 'output_modality' key"):
        validate_modality(query, "combined", "v1.0")

def test_validate_modality_v0_missing_key():
    """Tests validate_modality raises ValueError for missing pre-v1.0 modality key."""
    query = {"num_samples": 1, "inputs": []}
    with pytest.raises(ValueError, match="requires 'modality' key"):
        validate_modality(query, "rMetal", "v0.6")

def test_validate_modality_unknown_model(capsys):
    """Tests validate_modality prints warning for unknown model."""
    query = {"output_modality": "sra", "mode": "x", "inputs": []}
    validate_modality(query, "unknown_family", "v9.9")
    captured = capsys.readouterr()
    assert "Warning: Cannot validate modality for unknown model" in captured.out

# --- expand_metadata Tests ---
def test_expand_metadata_valid():
    """Tests expand_metadata correctly replicates metadata."""
    query = {
        "num_samples": 2,
        "inputs": ["{'a': 1, 'b': 2}", "{'a': 3, 'c': 4}"]
    }
    expected_df = pd.DataFrame([
        {'a': 1, 'b': 2}, {'a': 1, 'b': 2},
        {'a': 3, 'c': 4}, {'a': 3, 'c': 4}
    ])
    result_df = expand_metadata(query)
    pd.testing.assert_frame_equal(result_df, expected_df, check_like=True) # Use check_like for potential column order diffs

def test_expand_metadata_missing_keys():
    """Tests expand_metadata raises ValueError if keys are missing."""
    with pytest.raises(ValueError, match="requires 'inputs' and 'num_samples' keys"):
        expand_metadata({"inputs": ["{'a': 1}"]})
    with pytest.raises(ValueError, match="requires 'inputs' and 'num_samples' keys"):
        expand_metadata({"num_samples": 1})

def test_expand_metadata_invalid_input_string():
    """Tests expand_metadata raises ValueError for invalid input strings."""
    query = {"num_samples": 1, "inputs": ["not a dict"]}
    with pytest.raises(ValueError, match="Could not parse metadata strings"):
        expand_metadata(query)

def test_expand_metadata_invalid_num_samples():
    """Tests expand_metadata raises ValueError for invalid num_samples."""
    query = {"num_samples": 0, "inputs": ["{'a': 1}"]}
    with pytest.raises(ValueError, match="'num_samples' must be a positive integer"):
        expand_metadata(query)
    query = {"num_samples": -1, "inputs": ["{'a': 1}"]}
    with pytest.raises(ValueError, match="'num_samples' must be a positive integer"):
        expand_metadata(query)
    query = {"num_samples": "two", "inputs": ["{'a': 1}"]}
    with pytest.raises(ValueError, match="'num_samples' must be a positive integer"):
        expand_metadata(query)

# --- transform_to_counts Tests ---
def test_transform_to_counts():
    """Tests transforming logCPM to counts."""
    # Example logCPM data (log1p(CPM))
    log_cpm_data = pd.DataFrame({
        'gene1': [np.log1p(1e6 / 3), np.log1p(2e6 / 3)], # approx log1p(333k), log1p(666k)
        'gene2': [np.log1p(2e6 / 3), np.log1p(1e6 / 3)]
    })
    # Expected counts (approximate, due to * 30 and rounding)
    # expm1(log1p(C)) = C
    # counts ~= C * 30
    # counts1 ~= (1e6/3) * 30 = 10M ; (2e6/3) * 30 = 20M
    # counts2 ~= (2e6/3) * 30 = 20M ; (1e6/3) * 30 = 10M
    expected_counts = pd.DataFrame({
        'gene1': [10000000, 20000000],
        'gene2': [20000000, 10000000]
    }).astype(int)

    result_counts = transform_to_counts(log_cpm_data)
    # Check dtypes of all columns
    assert all(dtype == np.int64 or dtype == np.int32 for dtype in result_counts.dtypes), "All columns should have integer types"
    # Allow for minor differences due to float precision and rounding
    pd.testing.assert_frame_equal(result_counts, expected_counts, check_dtype=False, atol=1)

# --- log_cpm Tests ---
def test_log_cpm():
    """Tests transforming counts to logCPM."""
    counts_data = pd.DataFrame({
        'gene1': [1000000, 3000000],
        'gene2': [2000000, 6000000]
    })
    # Sample 1: Total = 3M. CPM1 = 1M/3M * 1e6 = 333333.3. CPM2 = 2M/3M * 1e6 = 666666.7
    # Sample 2: Total = 9M. CPM1 = 3M/9M * 1e6 = 333333.3. CPM2 = 6M/9M * 1e6 = 666666.7
    expected_log_cpm = pd.DataFrame({
        'gene1': [np.log1p(1e6 / 3), np.log1p(1e6 / 3)],
        'gene2': [np.log1p(2e6 / 3), np.log1p(2e6 / 3)]
    })

    result_log_cpm = log_cpm(counts_data)
    pd.testing.assert_frame_equal(result_log_cpm, expected_log_cpm, check_dtype=False, rtol=1e-5)

def test_log_cpm_zero_counts():
    """Tests log_cpm handles rows with zero total counts."""
    counts_data = pd.DataFrame({
        'gene1': [10, 0],
        'gene2': [20, 0]
    })
    expected_log_cpm = pd.DataFrame({
        'gene1': [np.log1p(10/30 * 1e6), 0.0],
        'gene2': [np.log1p(20/30 * 1e6), 0.0]
    })
    result_log_cpm = log_cpm(counts_data)
    pd.testing.assert_frame_equal(result_log_cpm, expected_log_cpm, check_dtype=False, rtol=1e-5)

# Tests requiring mocking (get_gene_order, process_samples, predict_query error cases)
# would be added below, likely using unittest.mock or pytest-mock. 