"""Ordinary hosted calls retain each GEM-2 operation's scientific results."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from pysynthbio.call_model_api import predict_query


@pytest.mark.parametrize("modality", ["bulk", "sc"])
@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize(
    "operation,suffix",
    [
        ("predict-expression", "baseline"),
        ("predict-expression-from-reference", "reference-conditioning"),
        ("predict-metadata", "predict-metadata"),
        ("autoencode", "autoencode"),
    ],
)
def test_ordinary_gem2_client_dispatch(operation, suffix, canonical, modality):
    row = {"counts": [16777217, 2], "latents": {"generated": [0.1, 0.2]}}
    if operation == "predict-expression-from-reference":
        row["latents"] = {"z_ref": [0.1], "z_target": [0.2], "z_displacement": [0.1]}
    elif operation == "predict-metadata":
        row = {
            "predicted_metadata": {
                "tissue_ontology_id": "UBERON:0002107",
                "age_years": None,
            },
            "classifier_probabilities": {
                "tissue_ontology_id": {"UBERON:0002107": 0.75}
            },
            "latents": {"encoded": [0.3, 0.4]},
        }
    elif operation == "autoencode":
        row = {
            "counts": [16777217, 2],
            "model_release_id": "qualified",
            "vae_run_id": "same-vae",
        }
    content = {
        "outputs": [row, deepcopy(row)],
        "gene_order": ["gene1", "gene2"],
        "model_version": "qualified",
    }
    original = deepcopy(content)
    model = "gem-2" if canonical else f"gem-2-{modality}_{suffix}"
    query = (
        {
            "operation": operation,
            "inputs": {
                "modality": "bulk" if modality == "bulk" else "single-cell",
                "items": [],
            },
        }
        if canonical
        else {"inputs": []}
    )
    module = "pysynthbio.call_model_api"
    with patch(module + ".has_synthesize_token", return_value=True), patch(
        module + "._start_model_query", return_value="id"
    ) as start, patch(
        module + "._poll_model_query",
        return_value=("ready", {"downloadUrl": "https://example.com/result"}),
    ), patch(
        module + ".get_json", return_value=content
    ):
        result = predict_query(query, model_id=model, auto_authenticate=False)
    assert start.call_args.kwargs["query"]["source"] == "pysynthbio"
    assert result["metadata"].shape[0] == 2
    assert result["latents"].shape[0] == 2
    if operation == "predict-metadata":
        assert (
            result["metadata"]["tissue_ontology_id"].tolist() == ["UBERON:0002107"] * 2
        )
        assert result["classifier_probs"].iloc[0, 0] == {"UBERON:0002107": 0.75}
        assert result["expression"].shape == (2, 0)
    else:
        assert result["expression"].values.tolist() == [[16777217, 2]] * 2
        assert result["expression"].columns.tolist() == ["gene1", "gene2"]
        if operation == "autoencode":
            assert (
                result["provenance"].to_dict("records")
                == [{"model_release_id": "qualified", "vae_run_id": "same-vae"}] * 2
            )
    assert content == original
