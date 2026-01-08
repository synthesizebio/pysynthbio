import os

import pytest

# these tests are meant to be run for local integration, not on CI

api_key_available = "SYNTHESIZE_API_KEY" in os.environ
skip_reason_api_key = "SYNTHESIZE_API_KEY environment variable not set"

base_url_available = "API_BASE_URL" in os.environ
skip_reason_base_url = "API_BASE_URL environment variable not set"


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
@pytest.mark.skipif(not base_url_available, reason=skip_reason_base_url)
def test_list_models():
    from pysynthbio.list_models import list_models

    models = list_models(api_base_url=os.environ.get("API_BASE_URL"))
    assert len(models["models"]) > 0
    assert all("model_id" in model for model in models["models"])


@pytest.mark.skipif(not api_key_available, reason=skip_reason_api_key)
@pytest.mark.skipif(not base_url_available, reason=skip_reason_base_url)
def test_model_integration():
    models = [
        {
            "model_id": "gem-1-sc",
            "expected_outputs": ["expression", "metadata"],
            "returns_list": False,
        },
        {
            "model_id": "gem-1-bulk",
            "expected_outputs": ["expression", "metadata"],
            "returns_list": False,
        },
        {
            "model_id": "gem-1-sc_predict-metadata",
            "expected_outputs": [
                "classifier_probs",
                "latents",
                "metadata",
                "decoder_sample",
            ],
            "returns_list": True,
        },
        {
            "model_id": "gem-1-bulk_predict-metadata",
            "expected_outputs": [
                "classifier_probs",
                "latents",
                "metadata",
                "decoder_sample",
            ],
            "returns_list": True,
        },
        {
            "model_id": "gem-1-bulk_reference-conditioning",
            "expected_outputs": ["expression", "metadata"],
            "returns_list": False,
        },
        {
            "model_id": "gem-1-sc_reference-conditioning",
            "expected_outputs": ["expression", "metadata"],
            "returns_list": False,
        },
    ]

    from pysynthbio.call_model_api import predict_query
    from pysynthbio.get_example_query import get_example_query

    for model in models:
        print(f"Testing model: {model['model_id']}")
        query = get_example_query(
            model_id=model["model_id"],
            api_base_url=os.environ.get("API_BASE_URL"),
        )
        result = predict_query(
            query=query["example_query"],
            model_id=model["model_id"],
            auto_authenticate=False,
            api_base_url=os.environ.get("API_BASE_URL"),
        )

        if model["returns_list"]:
            # Metadata prediction models return a list of output dicts
            assert isinstance(result, list), f"Expected list for {model['model_id']}"
            assert len(result) > 0, f"Expected non-empty list for {model['model_id']}"
            for output_key in model["expected_outputs"]:
                assert (
                    output_key in result[0]
                ), f"Missing {output_key} in {model['model_id']}"
        else:
            # Standard models return a dict
            for output_key in model["expected_outputs"]:
                assert (
                    output_key in result
                ), f"Missing {output_key} in {model['model_id']}"
