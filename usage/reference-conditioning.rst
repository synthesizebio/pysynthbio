Reference Conditioning
======================

Overview
--------

Reference conditioning models generate expression data **conditioned on a real reference sample**. This allows you to "anchor" to an existing expression profile while applying perturbations or modifications.

This is useful when you want to:

- Simulate the effect of a perturbation on a specific sample
- Generate expression profiles that preserve the biological and technical characteristics of a reference
- Create synthetic "treated vs. control" pairs

Available Models
----------------

- **gem-1-bulk_reference-conditioning**: Bulk RNA-seq reference conditioning model
- **gem-1-sc_reference-conditioning**: Single-cell RNA-seq reference conditioning model

.. note::
   These endpoints may require 1-2 minutes of startup time if they have been scaled down. Plan accordingly for interactive use.

.. code-block:: python

    import pysynthbio

How It Works
------------

Reference conditioning encodes the biological and technical characteristics from a real expression sample, then generates new expression data that:

1. Preserves the biological/technical latent space of the reference
2. Applies any perturbation metadata you specify
3. Returns synthetic expression that reflects the perturbation effect on that specific sample

Creating a Query
----------------

Reference conditioning queries require different inputs than baseline models:

.. code-block:: python

    # Get the example query structure
    example_query = pysynthbio.get_example_query(model_id="gem-1-bulk_reference-conditioning")["example_query"]

    # Inspect the query structure
    print(example_query)

The query structure includes:

1. **inputs**: A list where each input contains:

   - **counts**: The reference expression counts (a dictionary with a ``counts`` list)
   - **metadata**: Perturbation-only metadata (see below)
   - **num_samples**: How many samples to generate

2. **conditioning**: Which latent spaces to condition on (typically ``["biological", "technical"]``)

3. **sampling_strategy**: ``"mean estimation"`` or ``"sample generation"``

Perturbation-Only Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike baseline models, reference conditioning queries only accept perturbation metadata fields:

- ``perturbation_ontology_id``
- ``perturbation_type``
- ``perturbation_time``
- ``perturbation_dose``

All other biological and technical metadata is inferred from the reference expression.

Example: Simulating a Drug Treatment
------------------------------------

Here's a complete example simulating a drug treatment effect on a reference sample:

.. code-block:: python

    # Start with example query structure
    query = pysynthbio.get_example_query(model_id="gem-1-bulk_reference-conditioning")["example_query"]

    # Replace with your actual reference counts
    # The counts list must match the model's expected gene order and length
    query["inputs"][0]["counts"] = {"counts": your_reference_counts}

    # Specify the perturbation
    query["inputs"][0]["metadata"] = {
        "perturbation_ontology_id": "CHEMBL25",  # Aspirin (ChEMBL ID)
        "perturbation_type": "compound",
        "perturbation_time": "24h",
        "perturbation_dose": "10uM"
    }

    query["inputs"][0]["num_samples"] = 3

    # Set the sampling strategy
    query["sampling_strategy"] = "mean estimation"

    # Submit the query
    result = pysynthbio.predict_query(query, model_id="gem-1-bulk_reference-conditioning")

Example: CRISPR Knockout Simulation
-----------------------------------

Simulate the effect of knocking out a specific gene:

.. code-block:: python

    query = pysynthbio.get_example_query(model_id="gem-1-bulk_reference-conditioning")["example_query"]

    # Your reference sample counts
    query["inputs"][0]["counts"] = {"counts": control_sample_counts}

    # CRISPR knockout of TP53
    query["inputs"][0]["metadata"] = {
        "perturbation_ontology_id": "ENSG00000141510",  # TP53 Ensembl ID
        "perturbation_type": "crispr"
    }

    query["inputs"][0]["num_samples"] = 5

    result = pysynthbio.predict_query(query, model_id="gem-1-bulk_reference-conditioning")

Query Parameters
----------------

conditioning (list, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls which latent spaces are conditioned on the reference. Default is ``["biological", "technical"]``.

When both are conditioned, the model preserves both biological identity and technical characteristics from the reference sample.

sampling_strategy (str, required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls the type of prediction:

- **"sample generation"**: Generates realistic-looking synthetic data with measurement error. **(Bulk only)**
- **"mean estimation"**: Provides stable mean estimates. **(Bulk and single-cell)**

.. code-block:: python

    query["sampling_strategy"] = "mean estimation"

fixed_total_count (bool, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls whether to preserve the reference's library size:

- **False** (default): The output's total count is taken from the reference expression (sum of its counts). Use this when you want the synthetic sample to preserve the reference's library size.
- **True**: Forces the model to use the ``total_count`` parameter value (or default) instead of the reference's library size.

.. code-block:: python

    # Preserve reference library size (default)
    query["fixed_total_count"] = False

    # Or force a specific library size
    query["fixed_total_count"] = True
    query["total_count"] = 10000000

total_count (int, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Library size used when converting predicted log CPM back to raw counts. Only effective when ``fixed_total_count = True``.

- Default: 10,000,000 for bulk; 10,000 for single-cell

deterministic_latents (bool, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``True``, the model uses the mean of each latent distribution (``p(z|metadata)`` for perturbation, ``q(z|x)`` for conditioned components) instead of sampling. This produces deterministic, reproducible outputs.

- Default: ``False``

.. code-block:: python

    query["deterministic_latents"] = True

seed (int, optional)
^^^^^^^^^^^^^^^^^^^^

Random seed for reproducibility.

.. code-block:: python

    query["seed"] = 42

Valid Perturbation Metadata
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description / Format
   * - ``perturbation_ontology_id``
     - Ensembl gene ID (e.g., ``ENSG00000141510``), `ChEBI ID <https://www.ebi.ac.uk/chebi/>`_, `ChEMBL ID <https://www.ebi.ac.uk/chembl/>`_, or `NCBI Taxonomy ID <https://www.ncbi.nlm.nih.gov/taxonomy>`_
   * - ``perturbation_type``
     - One of: "coculture", "compound", "control", "crispr", "genetic", "infection", "other", "overexpression", "peptide or biologic", "shrna", "sirna"
   * - ``perturbation_time``
     - Time since perturbation (e.g., "24h", "48h")
   * - ``perturbation_dose``
     - Dose of perturbation (e.g., "10uM", "1mg/kg")

Working with Results
--------------------

The result structure is similar to baseline models:

.. code-block:: python

    # Access metadata and expression matrices
    metadata = result["metadata"]
    expression = result["expression"]

    # Compare to your reference
    print(expression.shape)
    print(metadata.head())

Differential Expression
^^^^^^^^^^^^^^^^^^^^^^^

When conditioning on both biological and technical latents, you can directly compare the generated expression to your reference to identify perturbation effects:

.. code-block:: python

    import numpy as np

    # Your reference (input) counts
    reference_cpm = your_reference_counts / np.sum(your_reference_counts) * 1e6

    # Generated (perturbed) counts
    generated_counts = expression.iloc[0].values
    generated_cpm = generated_counts / np.sum(generated_counts) * 1e6

    # Log fold change
    log2fc = np.log2(generated_cpm + 1) - np.log2(reference_cpm + 1)

    # Identify top changed genes
    gene_names = expression.columns
    top_indices = np.argsort(log2fc)[-20:]
    print("Top upregulated genes:", gene_names[top_indices].tolist())

Important Notes
---------------

Counts Vector Length
^^^^^^^^^^^^^^^^^^^^

The reference counts vector must match the model's expected number of genes. If the length doesn't match, the API will return a validation error.

Use ``get_example_query()`` to see the expected structure and ensure your counts vector has the correct length.

Gene Order
^^^^^^^^^^

Ensure your reference counts are in the same gene order expected by the model. The response includes a ``gene_order`` field that specifies the expected order.
