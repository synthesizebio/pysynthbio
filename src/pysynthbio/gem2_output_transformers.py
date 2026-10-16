"""Format the four hosted GEM-2 operations without dropping result fields."""

import re
from typing import Callable

import pandas as pd

from pysynthbio.output_transformers import transform_standard_model_output

_OPERATIONS = {
    "baseline": "predict-expression",
    "reference-conditioning": "predict-expression-from-reference",
    "metadata": "predict-metadata",
    "predict-metadata": "predict-metadata",
    "autoencode": "autoencode",
}
_ALIAS = re.compile(
    r"^gem-2(?:-(bulk|sc))?(?:_(baseline|reference-conditioning|predict-metadata|metadata|autoencode))?(?:-ensemble)?$"
)
_MEMBER = re.compile(
    r"^gem-2-(bulk|sc)_reference-conditioning-(4w6l3p61|p3rz5xv7|x4s7hdmu)$"
)


def transform_gem2_metadata(content: dict) -> dict:
    outputs = content["outputs"]
    return {
        "metadata": pd.DataFrame([row["predicted_metadata"] for row in outputs]),
        "latents": pd.DataFrame([row["latents"] for row in outputs]),
        "classifier_probs": pd.DataFrame(
            [row["classifier_probabilities"] for row in outputs]
        ),
        # GEM-2 metadata prediction does not generate expression counts.
        "expression": pd.DataFrame(index=range(len(outputs))),
    }


def transform_gem2_expression(content: dict) -> dict:
    # Preserve positions for absent/null metadata or optional embedding rows.
    rows = [
        {
            **row,
            "metadata": {} if row.get("metadata") is None else row["metadata"],
            "latents": {} if row.get("latents") is None else row["latents"],
        }
        for row in content["outputs"]
    ]
    return transform_standard_model_output({**content, "outputs": rows})


def transform_gem2_autoencode(content: dict) -> dict:
    result = transform_gem2_expression(content)
    result["provenance"] = pd.DataFrame(
        [
            {key: row[key] for key in ("model_release_id", "vae_run_id")}
            for row in content["outputs"]
        ]
    )
    return result


def gem2_output_transformer(model_id: str, query: dict) -> Callable | None:
    match = _ALIAS.fullmatch(model_id)
    if model_id == "gem-2":
        operation = query.get("operation")
    elif match and (match[1] or match[2]):
        operation = _OPERATIONS.get(match[2], "predict-expression")
    elif _MEMBER.fullmatch(model_id):
        operation = "predict-expression-from-reference"
    else:
        return None
    return {
        "predict-expression": transform_gem2_expression,
        "predict-expression-from-reference": transform_gem2_expression,
        "predict-metadata": transform_gem2_metadata,
        "autoencode": transform_gem2_autoencode,
    }.get(operation)
