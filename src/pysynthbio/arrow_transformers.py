"""Parse an Apache Arrow IPC stream from a self-hosted container into DataFrames.

A self-hosted model container returns predictions as an Arrow IPC stream whose
schema metadata carries ``request_type``, ``model_version`` and (for expression
request types) ``gene_order``. This module converts that stream into the same
``dict`` of ``pandas.DataFrame`` objects that the hosted-API JSON transformers
produce, so calling code is identical regardless of where the model runs.

The expression matrix is built directly from the Arrow ``counts`` column via
NumPy (no intermediate Python ``int`` objects), which keeps peak memory close
to the size of the matrix itself rather than several times larger.
"""

import json
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
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


def _read_table(source) -> "pa.Table":
    """Read an Arrow IPC stream from raw bytes or a readable file-like object."""
    pa, ipc = _require_pyarrow()
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = pa.BufferReader(pa.py_buffer(bytes(source)))
    reader = ipc.open_stream(source)
    return reader.read_all()


def _schema_metadata(table: "pa.Table") -> Dict[str, str]:
    raw = table.schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def _gene_order(metadata: Dict[str, str]) -> Optional[list]:
    if "gene_order" not in metadata:
        return None
    return json.loads(metadata["gene_order"])


def _counts_matrix(list_column, n_genes: Optional[int]) -> np.ndarray:
    """Convert an Arrow ``list<int>`` column (one list per sample) to a dense 2D
    NumPy array without materializing Python ints.

    Counts are kept as int64 (matching the hosted JSON path) so downstream
    integer aggregations -- pseudobulk sums, column totals across many samples --
    cannot silently overflow.
    """
    import pyarrow as pa

    array = (
        list_column.combine_chunks()
        if isinstance(list_column, pa.ChunkedArray)
        else list_column
    )
    n_rows = len(array)
    flat = array.flatten().to_numpy(zero_copy_only=False)

    if n_rows == 0:
        width = n_genes if n_genes is not None else 0
        return flat.reshape(0, width)

    width = n_genes if n_genes is not None else flat.size // n_rows
    if width == 0 or flat.size != n_rows * width:
        raise ValueError(
            "Arrow counts column is not rectangular: "
            f"{flat.size} values cannot reshape to ({n_rows}, {width}). "
            "Every sample must carry a full gene vector."
        )

    matrix = flat.reshape(n_rows, width)
    if matrix.dtype != np.int64:
        matrix = matrix.astype(np.int64, copy=False)
    return matrix


def _expression_frame(list_column, gene_order: Optional[list]) -> pd.DataFrame:
    n_genes = len(gene_order) if gene_order is not None else None
    matrix = _counts_matrix(list_column, n_genes)
    return pd.DataFrame(matrix, columns=gene_order)


def _struct_field(table: "pa.Table", column: str, field: str):
    """Return a child field array of a struct column (e.g. decoder_sample.counts)."""
    import pyarrow as pa

    col = table.column(column)
    array = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    return array.field(field)


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


def _transform_expression(table: "pa.Table", gene_order: Optional[list]) -> Dict:
    return {
        "metadata": pd.DataFrame(table.column("metadata").to_pylist()),
        "expression": _expression_frame(table.column("counts"), gene_order),
        "latents": _latents_frame(table),
    }


def _transform_predict_metadata(table: "pa.Table", gene_order: Optional[list]) -> Dict:
    return {
        "metadata": pd.DataFrame(table.column("metadata").to_pylist()),
        "latents": pd.DataFrame(table.column("latents").to_pylist()),
        "classifier_probs": pd.DataFrame(table.column("classifier_probs").to_pylist()),
        "expression": _expression_frame(
            _struct_field(table, "decoder_sample", "counts"), gene_order
        ),
    }


def transform_arrow_stream(source: Union[bytes, bytearray, memoryview, object]) -> Dict:
    """Convert an Arrow stream (bytes or readable file-like) into a dict of frames."""
    table = _read_table(source)
    metadata = _schema_metadata(table)
    gene_order = _gene_order(metadata)
    request_type = metadata.get("request_type", "baseline")

    if request_type == "predict_metadata":
        return _transform_predict_metadata(table, gene_order)
    return _transform_expression(table, gene_order)
