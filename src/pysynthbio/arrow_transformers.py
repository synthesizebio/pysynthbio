"""Parse an Apache Arrow IPC stream from a self-hosted container into DataFrames.

A self-hosted model container returns predictions as an Arrow IPC stream whose
schema metadata carries ``request_type``, ``model_version`` and (for expression
request types) ``gene_order``. This module converts that stream into the same
``dict`` of ``pandas.DataFrame`` objects that the hosted-API JSON transformers
produce, so calling code is identical regardless of where the model runs.
"""

import json
from typing import TYPE_CHECKING, Dict

import pandas as pd

if TYPE_CHECKING:
    import pyarrow as pa


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as err:  # pragma: no cover - exercised via install extras
        raise ImportError(
            "Reading from a self-hosted container requires pyarrow. "
            "Install it with: pip install 'pysynthbio[self_hosted]'"
        ) from err
    return pa, ipc


def _read_table(arrow_bytes: bytes) -> "pa.Table":
    pa, ipc = _require_pyarrow()
    reader = ipc.open_stream(pa.BufferReader(arrow_bytes))
    return reader.read_all()


def _schema_metadata(table: "pa.Table") -> Dict[str, str]:
    raw = table.schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def _gene_order(metadata: Dict[str, str]) -> list[str] | None:
    if "gene_order" not in metadata:
        return None
    return json.loads(metadata["gene_order"])


def _latents_frame(table: "pa.Table") -> pd.DataFrame:
    """Combine the per-key latent columns into biological/technical/perturbation."""
    columns = {
        "biological": "biological_latent",
        "technical": "technical_latent",
        "perturbation": "perturbation_latent",
    }
    data = {
        out_key: table.column(col).to_pylist()
        for out_key, col in columns.items()
        if col in table.schema.names
    }
    return pd.DataFrame(data)


def _expression_frame(counts_rows: list, gene_order: list[str] | None) -> pd.DataFrame:
    if gene_order is not None:
        expression = pd.DataFrame(counts_rows, columns=gene_order)
    else:
        expression = pd.DataFrame(counts_rows)
    return expression.astype(int)


def _transform_expression(table: "pa.Table", gene_order: list[str] | None) -> Dict:
    counts_rows = table.column("counts").to_pylist()
    metadata = pd.DataFrame(table.column("metadata").to_pylist())
    return {
        "metadata": metadata,
        "expression": _expression_frame(counts_rows, gene_order),
        "latents": _latents_frame(table),
    }


def _transform_predict_metadata(
    table: "pa.Table", gene_order: list[str] | None
) -> Dict:
    metadata = pd.DataFrame(table.column("metadata").to_pylist())
    latents = pd.DataFrame(table.column("latents").to_pylist())
    classifier_probs = pd.DataFrame(table.column("classifier_probs").to_pylist())
    decoder_rows = [
        row.get("counts", []) for row in table.column("decoder_sample").to_pylist()
    ]
    return {
        "metadata": metadata,
        "latents": latents,
        "classifier_probs": classifier_probs,
        "expression": _expression_frame(decoder_rows, gene_order),
    }


def transform_arrow_stream(arrow_bytes: bytes) -> Dict:
    """Convert Arrow stream bytes into a dict of DataFrames keyed by output type."""
    table = _read_table(arrow_bytes)
    metadata = _schema_metadata(table)
    gene_order = _gene_order(metadata)
    request_type = metadata.get("request_type", "baseline")

    if request_type == "predict_metadata":
        return _transform_predict_metadata(table, gene_order)
    return _transform_expression(table, gene_order)
