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

The structure of the query required by the API is fixed for the current supported model (combined v1.0). You can use `get_valid_query` to get a correctly structured example dictionary.

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
