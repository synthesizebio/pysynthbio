# pysynthbio

The Pythonic API calling package for Synthesize Bio

## Installation

To start using pysynthbio, first you need to have an account with synthesize.bio.
[Go here to create one](https://app.synthesize.bio/)

Then on your machine you can install using pip:

```
pip install pysynthbio
```

To ensure it installed you can run `pip show pysynthbio`.

### Installing from a GitHub Release

Alternatively, you can install a specific version directly from its GitHub Release page. This is useful for testing pre-releases or specific tagged versions.

1.  Go to the [Releases page](https://github.com/synthesizebio/pysynthbio/releases) of the repository.
2.  Find the release tag you want to install (e.g., `v2.0.0`).
3.  Expand the "Assets" section for that release.
4.  Download the `.whl` (wheel) file or the `.tar.gz` (source distribution) file. The wheel file is generally preferred if available for your platform.
5.  Install the downloaded file using pip, replacing `<path_to_downloaded_file>` with the actual path to the file:

    ```bash
    # Example using a downloaded wheel file
    pip install /path/to/pysynthbio-2.0.0-py3-none-any.whl

    # Example using a downloaded source distribution
    pip install /path/to/pysynthbio-2.0.0.tar.gz
    ```

## Usage

### Get your API key

Go to https://app.synthesize.bio/profile to generate an API key. Then set this key as an environment variable named `SYNTHESIZE_API_KEY` to authenticate your API requests.

### Form a request

First, import the necessary functions from the package:

```python
import pysynthbio
```

### Discover Valid Modalities

To see which output modalities are supported by the current model, use `get_valid_modalities`. This function returns a set of strings representing the allowed values for the `output_modality` key in your query.

```python
supported_modalities = pysynthbio.get_valid_modalities()
print(supported_modalities)
# Output might look like: {'bulk_rna-seq', 'lincs', 'sra', ...}
```

### Generate Example Queries

The structure of the query required by the API is fixed for the current supported model (v1.0). You can use `get_valid_query` to get a correctly structured example dictionary.

```python
# Get the example query structure
example_query = pysynthbio.get_valid_query()
```

### Get Predictions

Use `predict_query` to send a query to the API and get expression predictions. You'll typically use `get_valid_query` to help structure your request. This function also requires the API key.

```python
# You can modify the example_query or create your own following the structure
my_query = pysynthbio.get_valid_query() # Example: using the default valid query
# Modify my_query as needed...

results = pysynthbio.predict_query(
    query=my_query,
    as_counts=True # Get results as estimated counts (default). Set to False for logCPM.
)

# Access results:
metadata_df = results["metadata"]
expression_df = results["expression"]
```

This covers the basic workflow: understanding the required query structure and making predictions.
