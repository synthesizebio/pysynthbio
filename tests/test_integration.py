import os
import pprint

import pytest

api_key_available = "SYNTHESIZE_API_KEY" in os.environ
skip_reason_api_key = "SYNTHESIZE_API_KEY environment variable not set"


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_list_models():
    from pysynthbio.list_models import list_models

    models = list_models(api_base_url=os.environ.get("API_BASE_URL"))
    assert len(models["models"]) > 0
    assert all("model_id" in model for model in models["models"])


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
def test_model_integration():
    model_ids = [
        # "gem-1-bulk_reference-conditioning",
        "gem-1-sc",
    ]
    import pandas as pd

    from pysynthbio.call_model_api import predict_query
    from pysynthbio.get_example_query import get_example_query

    for model_id in model_ids:
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

        assert "metadata" in result
        assert "expression" in result
        # check that the expression is a DataFrame
        assert isinstance(result["expression"], pd.DataFrame)
        # check the length of the expression matches query
        print(
            len(result["expression"]),
            len(query["example_query"]["inputs"]),
        )
        pprint.pprint(query["example_query"]["inputs"])
        samples_per_input = query["example_query"]["inputs"][0]["num_samples"] * len(
            query["example_query"]["inputs"]
        )
        assert len(result["expression"]) == samples_per_input

        assert "latents" in result
        assert len(result["metadata"]) > 0
        assert len(result["expression"]) > 0
