"""Unit tests for output transformers and raw_response parameter."""

from unittest.mock import patch

import pandas as pd
import pytest

from pysynthbio.output_transformers import (
    OUTPUT_TRANSFORMERS,
    transform_metadata_model_output,
    transform_standard_model_output,
)

EXPECTED_MODEL_IDS = {
    "gem-1-bulk",
    "gem-1-sc",
    "gem-1-bulk_reference-conditioning",
    "gem-1-sc_reference-conditioning",
    "gem-1-bulk_condition-on-sample-ids",
    "gem-1-bulk_predict-metadata",
    "gem-1-sc_predict-metadata",
}


def test_all_model_ids_registered():
    assert set(OUTPUT_TRANSFORMERS.keys()) == EXPECTED_MODEL_IDS


def test_standard_models_use_standard_transformer():
    standard_ids = [
        "gem-1-bulk",
        "gem-1-sc",
        "gem-1-bulk_reference-conditioning",
        "gem-1-sc_reference-conditioning",
        "gem-1-bulk_condition-on-sample-ids",
    ]
    for model_id in standard_ids:
        assert OUTPUT_TRANSFORMERS[model_id] is transform_standard_model_output


def test_metadata_models_use_metadata_transformer():
    metadata_ids = [
        "gem-1-bulk_predict-metadata",
        "gem-1-sc_predict-metadata",
    ]
    for model_id in metadata_ids:
        assert OUTPUT_TRANSFORMERS[model_id] is transform_metadata_model_output


FAKE_STANDARD_JSON = {
    "gene_order": ["GENE1", "GENE2"],
    "outputs": [
        {"counts": [10, 20], "metadata": {"tissue": "brain"}, "latents": {"biological": [0.1], "technical": [0.2], "perturbation": [0.3]}},
    ],
}

FAKE_METADATA_JSON = {
    "outputs": [
        {"classifier_probs": {"tissue": {"brain": 0.9}}, "latents": {"biological": [0.1], "technical": [0.2], "perturbation": [0.3]}, "metadata": {"tissue": "brain"}, "decoder_sample": {"counts": [10, 20]}},
        {"classifier_probs": {"tissue": {"liver": 0.8}}, "latents": {"biological": [0.4], "technical": [0.5], "perturbation": [0.6]}, "metadata": {"tissue": "liver"}, "decoder_sample": {"counts": [30, 40]}},
    ],
}


def test_metadata_transformer_returns_dataframes():
    result = transform_metadata_model_output(FAKE_METADATA_JSON)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"metadata", "latents", "classifier_probs", "expression"}

    assert isinstance(result["metadata"], pd.DataFrame)
    assert len(result["metadata"]) == 2
    assert "tissue" in result["metadata"].columns

    assert isinstance(result["latents"], pd.DataFrame)
    assert len(result["latents"]) == 2
    assert set(result["latents"].columns) == {"biological", "technical", "perturbation"}

    assert isinstance(result["classifier_probs"], pd.DataFrame)
    assert len(result["classifier_probs"]) == 2
    assert isinstance(result["classifier_probs"]["tissue"].iloc[0], dict)

    assert isinstance(result["expression"], pd.DataFrame)
    assert len(result["expression"]) == 2
    assert result["expression"].dtypes.apply(lambda dt: pd.api.types.is_integer_dtype(dt)).all()
    assert result["expression"].iloc[0].tolist() == [10, 20]


def test_metadata_transformer_uses_gene_order_when_present():
    json_with_genes = {**FAKE_METADATA_JSON, "gene_order": ["GENE1", "GENE2"]}
    result = transform_metadata_model_output(json_with_genes)

    assert list(result["expression"].columns) == ["GENE1", "GENE2"]


def test_standard_transformer_returns_dataframes():
    result = transform_standard_model_output(FAKE_STANDARD_JSON)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"metadata", "expression", "latents"}
    assert isinstance(result["expression"], pd.DataFrame)
    assert list(result["expression"].columns) == ["GENE1", "GENE2"]
    assert result["expression"].iloc[0].tolist() == [10, 20]


MOCK_PREDICT_QUERY = "pysynthbio.call_model_api"


@patch(f"{MOCK_PREDICT_QUERY}.get_json")
@patch(f"{MOCK_PREDICT_QUERY}._poll_model_query")
@patch(f"{MOCK_PREDICT_QUERY}._start_model_query")
@patch(f"{MOCK_PREDICT_QUERY}.has_synthesize_token", return_value=True)
def test_raw_response_returns_unformatted_json(
    _mock_token, mock_start, mock_poll, mock_get_json
):
    from pysynthbio.call_model_api import predict_query

    mock_start.return_value = "query-123"
    mock_poll.return_value = ("ready", {"downloadUrl": "https://example.com/results.json"})
    mock_get_json.return_value = FAKE_STANDARD_JSON

    result = predict_query(
        query={"inputs": []},
        model_id="gem-1-bulk",
        auto_authenticate=False,
        raw_response=True,
    )

    assert result is FAKE_STANDARD_JSON
    assert isinstance(result, dict)
    assert "gene_order" in result


@patch(f"{MOCK_PREDICT_QUERY}.get_json")
@patch(f"{MOCK_PREDICT_QUERY}._poll_model_query")
@patch(f"{MOCK_PREDICT_QUERY}._start_model_query")
@patch(f"{MOCK_PREDICT_QUERY}.has_synthesize_token", return_value=True)
def test_unregistered_model_raises_without_raw_response(
    _mock_token, mock_start, mock_poll, mock_get_json
):
    from pysynthbio.call_model_api import predict_query

    mock_start.return_value = "query-123"
    mock_poll.return_value = ("ready", {"downloadUrl": "https://example.com/results.json"})
    mock_get_json.return_value = {"some": "data"}

    with pytest.raises(ValueError, match="No output formatter registered"):
        predict_query(
            query={"inputs": []},
            model_id="gem-1-nonexistent",
            auto_authenticate=False,
        )


@patch(f"{MOCK_PREDICT_QUERY}.get_json")
@patch(f"{MOCK_PREDICT_QUERY}._poll_model_query")
@patch(f"{MOCK_PREDICT_QUERY}._start_model_query")
@patch(f"{MOCK_PREDICT_QUERY}.has_synthesize_token", return_value=True)
def test_unregistered_model_works_with_raw_response(
    _mock_token, mock_start, mock_poll, mock_get_json
):
    from pysynthbio.call_model_api import predict_query

    fake_json = {"some": "data"}
    mock_start.return_value = "query-123"
    mock_poll.return_value = ("ready", {"downloadUrl": "https://example.com/results.json"})
    mock_get_json.return_value = fake_json

    result = predict_query(
        query={"inputs": []},
        model_id="gem-1-nonexistent",
        auto_authenticate=False,
        raw_response=True,
    )

    assert result is fake_json


@patch(f"{MOCK_PREDICT_QUERY}.get_json")
@patch(f"{MOCK_PREDICT_QUERY}._poll_model_query")
@patch(f"{MOCK_PREDICT_QUERY}._start_model_query")
@patch(f"{MOCK_PREDICT_QUERY}.has_synthesize_token", return_value=True)
def test_registered_model_applies_transformer(
    _mock_token, mock_start, mock_poll, mock_get_json
):
    from pysynthbio.call_model_api import predict_query

    mock_start.return_value = "query-123"
    mock_poll.return_value = ("ready", {"downloadUrl": "https://example.com/results.json"})
    mock_get_json.return_value = FAKE_STANDARD_JSON

    result = predict_query(
        query={"inputs": []},
        model_id="gem-1-bulk",
        auto_authenticate=False,
    )

    assert isinstance(result, dict)
    assert "expression" in result
    assert "metadata" in result
    assert "latents" in result


@patch(f"{MOCK_PREDICT_QUERY}.get_json")
@patch(f"{MOCK_PREDICT_QUERY}._poll_model_query")
@patch(f"{MOCK_PREDICT_QUERY}._start_model_query")
@patch(f"{MOCK_PREDICT_QUERY}.has_synthesize_token", return_value=True)
def test_metadata_model_applies_transformer_via_predict_query(
    _mock_token, mock_start, mock_poll, mock_get_json
):
    from pysynthbio.call_model_api import predict_query

    mock_start.return_value = "query-123"
    mock_poll.return_value = ("ready", {"downloadUrl": "https://example.com/results.json"})
    mock_get_json.return_value = FAKE_METADATA_JSON

    result = predict_query(
        query={"inputs": []},
        model_id="gem-1-bulk_predict-metadata",
        auto_authenticate=False,
    )

    assert isinstance(result, dict)
    assert "expression" in result
    assert "metadata" in result
    assert "latents" in result
    assert "classifier_probs" in result
    assert isinstance(result["expression"], pd.DataFrame)
    assert isinstance(result["classifier_probs"], pd.DataFrame)
