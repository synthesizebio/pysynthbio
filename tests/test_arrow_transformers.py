"""Tests for parsing a self-hosted container's Arrow stream into DataFrames.

The fixtures here mirror exactly what the inference container emits (see
``inference_container/app/arrow_streaming.py``): an Arrow IPC stream whose schema
metadata carries ``request_type``, ``model_version`` and ``gene_order``.
"""

import json

import pyarrow as pa
import pyarrow.ipc as ipc

from pysynthbio.arrow_transformers import transform_arrow_stream

GENE_ORDER = ["ENSG1", "ENSG2", "ENSG3"]


def _to_stream(batch: pa.RecordBatch, metadata: dict[bytes, bytes]) -> bytes:
    schema = batch.schema.with_metadata(metadata)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def _baseline_stream() -> bytes:
    batch = pa.RecordBatch.from_pylist(
        [
            {
                "counts": [1, 2, 3],
                "metadata": {"tissue_ontology_id": "UBERON:0002107", "sex": "male"},
                "biological_latent": [0.1, 0.2],
                "technical_latent": [0.3, 0.4],
                "perturbation_latent": [],
            },
            {
                "counts": [4, 5, 6],
                "metadata": {"tissue_ontology_id": "UBERON:0002107", "sex": "female"},
                "biological_latent": [0.5, 0.6],
                "technical_latent": [0.7, 0.8],
                "perturbation_latent": [],
            },
        ]
    )
    metadata = {
        b"model_version": b"2.2",
        b"request_type": b"baseline",
        b"gene_order": json.dumps(GENE_ORDER).encode(),
    }
    return _to_stream(batch, metadata)


def test_baseline_stream_to_frames():
    result = transform_arrow_stream(_baseline_stream())

    assert set(result) == {"metadata", "expression", "latents"}
    expression = result["expression"]
    assert list(expression.columns) == GENE_ORDER
    assert expression.iloc[0].tolist() == [1, 2, 3]
    assert str(expression.dtypes.iloc[0]).startswith("int")

    assert result["metadata"].iloc[1]["sex"] == "female"
    assert list(result["latents"].columns) == [
        "biological",
        "technical",
        "perturbation",
    ]
    assert result["latents"].iloc[0]["technical"] == [0.3, 0.4]


def test_predict_metadata_stream_to_frames():
    batch = pa.RecordBatch.from_pylist(
        [
            {
                "classifier_probs": {"sex": {"male": 0.9, "female": 0.1}},
                "latents": {
                    "biological": [0.1],
                    "technical": [0.2],
                    "perturbation": [],
                },
                "metadata": {"sex": "male"},
                "decoder_sample": {"counts": [1, 2, 3]},
            }
        ]
    )
    metadata = {b"model_version": b"2.2", b"request_type": b"predict_metadata"}
    stream = _to_stream(batch, metadata)

    result = transform_arrow_stream(stream)
    assert set(result) == {"metadata", "latents", "classifier_probs", "expression"}
    assert result["expression"].iloc[0].tolist() == [1, 2, 3]
    assert result["metadata"].iloc[0]["sex"] == "male"
