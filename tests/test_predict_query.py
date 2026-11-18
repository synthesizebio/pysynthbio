"""predict_query sends kwargs with the request without changing the caller's query."""

from unittest.mock import patch

from pysynthbio.call_model_api import predict_query

MODULE = "pysynthbio.call_model_api"
CONTENT = {
    "outputs": [{"counts": [1, 2], "metadata": {}}],
    "gene_order": ["gene1", "gene2"],
}


def test_kwargs_do_not_leak_into_later_calls():
    query = {"inputs": [{"metadata": {"tissue": "liver"}, "num_samples": 1}]}
    original = {"inputs": [{"metadata": {"tissue": "liver"}, "num_samples": 1}]}
    with patch(MODULE + ".has_synthesize_token", return_value=True), patch(
        MODULE + "._start_model_query", return_value="id"
    ) as start, patch(
        MODULE + "._poll_model_query",
        return_value=("ready", {"downloadUrl": "https://example.com/result"}),
    ), patch(
        MODULE + ".get_json", return_value=CONTENT
    ):
        predict_query(query, model_id="gem-1-bulk", deterministic_latents=True)
        predict_query(query, model_id="gem-1-bulk")

    first, second = (call.kwargs["query"] for call in start.call_args_list)
    assert first["deterministic_latents"] is True
    assert first["source"] == "pysynthbio"
    assert "deterministic_latents" not in second
    assert second["source"] == "pysynthbio"
    assert query == original
