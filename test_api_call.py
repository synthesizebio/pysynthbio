# test_api_call.py
import os
import sys
import pandas as pd
# from dotenv import load_dotenv # Keep commented out unless needed

# Load environment variables from .env file
# load_dotenv()

# Add src directory to path to allow import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    # Import with updated names/signatures
    from call_model_api import (
        predict_query,
        get_valid_query,
        get_valid_modalities,
        DEFAULT_MODEL_FAMILY, # Use new defaults
        DEFAULT_MODEL_VERSION
    )
except ImportError as e:
    print(f"Error importing functions: {e}")
    print("Ensure you are running this script from the workspace root directory.")
    sys.exit(1)

# --- Configuration ---
MODEL_FAMILY_TO_TEST = DEFAULT_MODEL_FAMILY
MODEL_VERSION_TO_TEST = DEFAULT_MODEL_VERSION
# Example for older model:
# MODEL_FAMILY_TO_TEST = "rMetal"
# MODEL_VERSION_TO_TEST = "v0.6"
# --- End Configuration ---

print(f"--- Testing model: {MODEL_FAMILY_TO_TEST}{MODEL_VERSION_TO_TEST} ---")

# Check for API Key
if "SYNTHESIZE_API_KEY" not in os.environ:
    print("Error: SYNTHESIZE_API_KEY environment variable is not set.")
    print("Please set the environment variable before running the test.")
    print("Example: export SYNTHESIZE_API_KEY='your_key_here'")
    sys.exit(1)
else:
    print("SYNTHESIZE_API_KEY found.")

try:
    # 1. Get a valid query structure for the model
    print(f"\n1. Generating valid query for {MODEL_FAMILY_TO_TEST} {MODEL_VERSION_TO_TEST}...")
    sample_query = get_valid_query(MODEL_FAMILY_TO_TEST, MODEL_VERSION_TO_TEST)
    print("Sample query generated:")
    print(sample_query)

    # 2. Call the predict_query function
    print(f"\n2. Calling predict_query for {MODEL_FAMILY_TO_TEST} {MODEL_VERSION_TO_TEST}...")
    results = predict_query(
        query=sample_query,
        model_family=MODEL_FAMILY_TO_TEST,
        model_version=MODEL_VERSION_TO_TEST,
        as_counts=True
    )
    print("API call successful.")

    # 3. Print results summary
    print("\n3. Results Summary:")
    metadata_df = results.get("metadata")
    expression_df = results.get("expression")

    if isinstance(metadata_df, pd.DataFrame):
        print(f"\nMetadata DataFrame shape: {metadata_df.shape}")
        print("Metadata DataFrame head:")
        print(metadata_df.head())
    else:
        print("\nMetadata not found or not a DataFrame.")

    if isinstance(expression_df, pd.DataFrame):
        print(f"\nExpression DataFrame shape: {expression_df.shape}")
        print("Expression DataFrame head:")
        print(expression_df.head())
    else:
        print("\nExpression data not found or not a DataFrame.")

except KeyError as e:
    print(f"\n--- Test Failed ---")
    print(f"KeyError: {e}")
    print("This likely means the SYNTHESIZE_API_KEY environment variable is missing.")
except ValueError as e:
    print(f"\n--- Test Failed ---")
    print(f"ValueError: {e}")
    print("This could be due to an API error, invalid model name/version, incorrect API key, or response issue.")
except Exception as e:
    print(f"\n--- Test Failed ---")
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Test Complete ---") 