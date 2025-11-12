import os

import pytest

api_key_available = "SYNTHESIZE_API_KEY" in os.environ
skip_reason_api_key = "SYNTHESIZE_API_KEY environment variable not set"

models = [
    "gem-1-bulk_predict-metadata",
    "gem-1-sc_predict-metadata",
]


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_model_integration():
    from pysynthbio.call_model_api import predict_query
    from pysynthbio.get_example_query import get_example_query

    for model_id in models:
        print(f"Testing model: {model_id}")
        query = get_example_query(
            model_id=model_id,
            api_base_url=os.environ.get("API_BASE_URL"),
        )
        result = predict_query(
            query=query["example_query"],
            model_id=model_id,
            auto_authenticate=False,
            api_base_url=os.environ.get("API_BASE_URL"),
        )

        # lets inspect the top level keys
        assert "classifier_probs" in result
        assert "latents" in result
        assert "metadata" in result
        assert len(result["metadata"]) > 0
        assert len(result["classifier_probs"]) > 0
        assert len(result["latents"]) > 0
