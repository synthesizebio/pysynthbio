Metadata Prediction
===================

Overview
--------

Metadata prediction models **infer biological metadata from observed expression data**. Given a gene expression profile, the model predicts the likely biological characteristics such as cell type, tissue, disease state, and more.

This is useful when you want to:

- Annotate samples of unknown origin
- Validate sample labels against expression patterns
- Discover potential mislabeled or contaminated samples
- Understand the biological characteristics captured in expression data

Available Models
----------------

- **gem-1-bulk_predict-metadata**: Bulk RNA-seq metadata prediction model
- **gem-1-sc_predict-metadata**: Single-cell RNA-seq metadata prediction model

.. note::
   These endpoints may require 1-2 minutes of startup time if they have been scaled down. Plan accordingly for interactive use.

.. code-block:: python

    import pysynthbio

How It Works
------------

Metadata prediction encodes your expression data into the model's latent space and then uses classifiers to predict the most likely metadata values for each sample. The model returns:

1. **Classifier probabilities**: For each categorical metadata field, the probability distribution over possible values
2. **Predicted labels**: The most likely value for each metadata field
3. **Latent representations**: The biological, technical, and perturbation latent vectors

Creating a Query
----------------

Metadata prediction queries are simpler than other model types—you only need to provide expression counts:

.. code-block:: python

    # Get the example query structure
    example_query = pysynthbio.get_example_query(model_id="gem-1-bulk_predict-metadata")

    # Inspect the query structure
    print(example_query)

The query structure includes:

1. **inputs**: A list of count vectors, where each element is a dictionary with a ``counts`` field containing expression values

2. **seed** (optional): Random seed for reproducibility

Example: Predicting Sample Metadata
-----------------------------------

Here's a complete example predicting metadata for expression samples:

.. code-block:: python

    # Start with example query structure
    query = pysynthbio.get_example_query(model_id="gem-1-bulk_predict-metadata")

    # Replace with your actual expression counts
    # Each input should be a dictionary with a counts list
    query["inputs"] = [
        {"counts": sample1_counts},
        {"counts": sample2_counts},
        {"counts": sample3_counts}
    ]

    # Optional: set seed for reproducibility
    query["seed"] = 42

    # Submit the query
    result = pysynthbio.predict_query(query, model_id="gem-1-bulk_predict-metadata")

Example: Single Sample Prediction
---------------------------------

For predicting metadata of a single sample:

.. code-block:: python

    query = pysynthbio.get_example_query(model_id="gem-1-bulk_predict-metadata")

    # Single sample
    query["inputs"] = [
        {"counts": my_sample_counts}
    ]

    result = pysynthbio.predict_query(query, model_id="gem-1-bulk_predict-metadata")

    # Access the predictions
    print(result["metadata"])

Query Parameters
----------------

inputs (list, required)
^^^^^^^^^^^^^^^^^^^^^^^

A list of expression count vectors. Each element should be a dictionary containing:

- **counts**: A list of non-negative integers representing gene expression counts

.. code-block:: python

    query["inputs"] = [
        {"counts": [0, 12, 5, 0, 33, 7, ...]},  # Sample 1
        {"counts": [3, 0, 0, 7, 1, 0, ...]}     # Sample 2
    ]

seed (int, optional)
^^^^^^^^^^^^^^^^^^^^

Random seed for reproducibility.

.. code-block:: python

    query["seed"] = 123

Understanding the Results
-------------------------

The results from metadata prediction include several components:

Predicted Metadata
^^^^^^^^^^^^^^^^^^

The ``metadata`` DataFrame contains the predicted values for each sample:

.. code-block:: python

    # View predicted metadata
    print(result["metadata"].head())

    # Access specific predictions
    print(result["metadata"]["cell_type_ontology_id"])
    print(result["metadata"]["tissue_ontology_id"])
    print(result["metadata"]["disease_ontology_id"])

Classifier Probabilities
^^^^^^^^^^^^^^^^^^^^^^^^

For categorical metadata fields, the model returns probability distributions over all possible values. These are useful for understanding prediction confidence:

.. code-block:: python

    # If probabilities are included in the output
    # Access cell type probabilities for first sample
    # The exact structure depends on the API response format

    # Example: viewing top predicted cell types
    if "classifier_probs" in result:
        cell_type_probs = result["classifier_probs"]["cell_type"][0]
        sorted_probs = sorted(cell_type_probs.items(), key=lambda x: x[1], reverse=True)
        print("Top predicted cell types:", sorted_probs[:5])

Latent Representations
^^^^^^^^^^^^^^^^^^^^^^

The model also returns latent vectors that capture biological, technical, and perturbation characteristics:

.. code-block:: python

    # Access latent representations (if returned)
    if "latents" in result:
        biological_latents = result["latents"]["biological"]
        technical_latents = result["latents"]["technical"]

Use Cases
---------

Sample Annotation
^^^^^^^^^^^^^^^^^

Annotate unlabeled samples with predicted metadata:

.. code-block:: python

    import pandas as pd

    # Load your unlabeled samples
    unlabeled_counts = pd.read_csv("unlabeled_samples.csv", index_col=0)

    # Create query
    query = pysynthbio.get_example_query(model_id="gem-1-bulk_predict-metadata")
    query["inputs"] = [
        {"counts": unlabeled_counts.iloc[:, i].tolist()}
        for i in range(unlabeled_counts.shape[1])
    ]

    # Predict metadata
    result = pysynthbio.predict_query(query, model_id="gem-1-bulk_predict-metadata")

    # Combine with sample IDs
    annotations = result["metadata"].copy()
    annotations["sample_id"] = unlabeled_counts.columns.tolist()

Quality Control
^^^^^^^^^^^^^^^

Validate existing sample labels against predicted metadata:

.. code-block:: python

    # Compare predicted vs. provided labels
    provided_labels = ["UBERON:0002107", "UBERON:0002107", "UBERON:0000955", "UBERON:0000955"]
    predicted_labels = result["metadata"]["tissue_ontology_id"].tolist()

    # Identify potential mismatches
    mismatches = [
        i for i, (p, pred) in enumerate(zip(provided_labels, predicted_labels))
        if p != pred
    ]
    if mismatches:
        print(f"Potential mislabeled samples: {mismatches}")

Batch Characterization
^^^^^^^^^^^^^^^^^^^^^^

Understand batch-specific technical characteristics:

.. code-block:: python

    import numpy as np

    # Group samples by batch
    batch_labels = ["batch1", "batch1", "batch2", "batch2"]

    # Check if technical predictions cluster by batch
    # This can help identify batch effects
    if "latents" in result:
        technical = result["latents"]["technical"]
        for batch in set(batch_labels):
            batch_indices = [i for i, b in enumerate(batch_labels) if b == batch]
            batch_mean = np.mean([technical[i][0] for i in batch_indices])
            print(f"{batch} technical latent mean: {batch_mean}")

Important Notes
---------------

Counts Vector Length
^^^^^^^^^^^^^^^^^^^^

The counts vector for each sample must match the model's expected number of genes. If the length doesn't match, the API will return a validation error.

Use ``get_example_query()`` to see the expected structure.

Gene Order
^^^^^^^^^^

Ensure your counts are in the same gene order expected by the model. The gene order should match what the baseline model expects—you can retrieve this from any prediction result's ``gene_order`` field.

Non-Negative Counts
^^^^^^^^^^^^^^^^^^^

All count values must be non-negative integers. Floats that are whole numbers (like ``10.0``) are accepted, but negative values will cause validation errors.
