# <img src="https://assets.synthesize.bio/logomark.png" style="width: 30px; height: 30px;" alt="Logomark">&nbsp;pysynthbio

`pysynthbio` is an Python package that provides a convenient interface to the [Synthesize Bio](https://www.synthesize.bio/) API, allowing users to generate realistic gene expression data based on specified biological conditions. This package enables researchers to easily access AI-generated transcriptomic data for various modalities including bulk RNA-seq and single-cell RNA-seq.

If you'd prefer 1-click dataset generation and analysis, try our [web platform](https://app.synthesize.bio/datasets/).

## Prerequisites

Create a [Synthesize Bio](https://app.synthesize.bio/) account and generate an [API Key](https://app.synthesize.bio/account/api-keys).

## Installation

```
pip install pysynthbio
```

## Quickstart

```python
import pysynthbio

# Set your API token (or use environment variable SYNTHESIZE_API_KEY)
pysynthbio.set_synthesize_token(use_keyring=True)

# Get an example query for the bulk RNA-seq baseline model
query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]

# Generate synthetic expression data
result = pysynthbio.predict_query(query, model_id="gem-1-bulk")

# Access the results
metadata = result["metadata"]
expression = result["expression"]
```

## Documentation

The canonical pysynthbio docs live at [docs.synthesize.bio/pysynthbio](https://docs.synthesize.bio/pysynthbio).

- [Get started](https://docs.synthesize.bio/pysynthbio/getting-started)
- [Installation](https://docs.synthesize.bio/pysynthbio/installation)
- [Models](https://docs.synthesize.bio/pysynthbio/models/baseline)

The legacy GitHub Pages site at `synthesizebio.github.io/pysynthbio` redirects to the new URLs and is no longer updated. The MDX source for the customer-facing docs lives in [`docs-external/`](./docs-external) and is aggregated into [`docs-external`](https://github.com/synthesizebio/docs-external) by the Mintlify multirepo workflow.

## Questions? Suggestions? Support?

Email us at [support@synthesize.bio](mailto:support@synthesize.bio). We'd love to hear from you!

## VERDICT models and trials

Sign in to [VERDICT API docs](https://verdict.synthesize.bio/app/api-docs/python)
for the Python client guide, API keys, model calls, and trial workflows.
An approved VERDICT account is required.

## Hosted GEM-2 operations

Use `predict_query(..., model_id="gem-2")` for a release that supports the requested
operation. GEM-2 uses `operation` to select `predict-expression`,
`predict-expression-from-reference`, `predict-metadata`, or `autoencode`.
The server checks support for the selected release and modality.

```python
import pysynthbio

query = {
    "operation": "predict-expression",
    "inputs": {
        "modality": "bulk",
        "items": [{
            "metadata": {
                "sample_type": "primary tissue",
                "library_layout": "PAIRED",
                "library_selection": "cDNA",
                "tissue": "liver",
                "bioactive_entity_perturbations": [{"smiles": "CCO"}],
                "gene_perturbations": [],
                "infection_perturbations": [],
            },
            "num_samples": 1,
            "output_total": {"type": "custom", "value": 8478},
        }],
    },
    "options": {"seed": 42, "sampling_strategy": "mean estimation"},
}
result = pysynthbio.predict_query(query, model_id="gem-2")
```

SMILES and direct ChEMBL IDs require an enabled molecule release. For a ChEMBL ID,
replace `{"smiles": "CCO"}` with `{"entity_ontology_id": "CHEMBL3261331"}`.
Supply exactly one of `entity`, `entity_ontology_id`, or `smiles` per molecule;
`entity` retains the supported-name lookup. Pin a release with the optional
request field `release` when testing a candidate.

Expression results contain `expression`, `metadata`, and `latents` DataFrames.
Metadata prediction contains predicted `metadata`, `classifier_probs`, and
`latents` with the `encoded` column; its `expression` has one row per result and
no columns because this operation does not generate counts. Reconstruction
(`autoencode`) retains counts and returns a `provenance` DataFrame with each
row's `model_release_id` and `vae_run_id`. Its `latents` has no columns.
Existing GEM-2 compatibility model names also use these operation-aware formats.
`raw_response=True` continues to return the original response unchanged.
