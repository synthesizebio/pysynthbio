# <img src="https://assets.synthesize.bio/logomark.png" style="width: 30px; height: 30px;" alt="Logomark">&nbsp;pysynthbio

`pysynthbio` is an Python package that provides a convenient interface to the [Synthesize Bio](https://www.synthesize.bio/) API, allowing users to generate realistic gene expression data based on specified biological conditions. This package enables researchers to easily access AI-generated transcriptomic data for various modalities including bulk RNA-seq and single-cell RNA-seq.

If you'd prefer 1-click dataset generation and analysis, try our [web platform](https://app.synthesize.bio/datasets/).

## Prerequisites

Create a [Synthesize Bio](https://app.synthesize.bio/) account and generate an [API Key](https://app.synthesize.bio/account/api-key).

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

[Get started](https://synthesizebio.mintlify.app/python-sdk/getting-started) | [Full Python SDK docs](https://synthesizebio.mintlify.app/python-sdk)

The user-facing documentation source lives in [`docs/`](./docs) and is published to [synthesizebio.mintlify.app](https://synthesizebio.mintlify.app) via the [`Sync docs to mintlify-docs`](./.github/workflows/sync-docs-to-mintlify.yml) workflow. See [`docs/README.md`](./docs/README.md) for contributor instructions.

## Questions? Suggestions? Support?

Email us at [support@synthesize.bio](mailto:support@synthesize.bio). We'd love to hear from you!
