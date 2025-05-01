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

### Discover Available Models

To see which models are currently available through the API, use `get_available_models`. This requires your API key environment variable to be set.

```python
available_models = pysynthbio.get_available_models()
```

### Generate Example Queries

The structure of the query required by the API depends on the model version. You can use `get_valid_query` to get a correctly structured example dictionary for a specific model name and version.

```python
# Example for a pre-v1.0 model (e.g., rMetal v0.6)
example_query_v0 = pysynthbio.get_valid_query('rMetal', 'v0.6')

# Example for a v1.0 model (e.g., combined v1.0)
example_query_v1 = pysynthbio.get_valid_query('combined', 'v1.0')
```

### Get Predictions

Use `predict_query` to send a query to a specific model and get expression predictions. You'll typically use `get_valid_query` to help structure your request. This function also requires the API key.

```python
query_structure = pysynthbio.get_valid_query('combined', 'v1.0')

results = pysynthbio.predict_query(
    query=my_query,
    model_name=target_model_name,
    model_version=target_model_version,
    as_counts=True # Get results as estimated counts (default). Set to False for logCPM.
)
```

This covers the basic workflow: discovering models, understanding query structures, and making predictions.
