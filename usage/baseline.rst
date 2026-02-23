Baseline Models
===============

Overview
--------

Baseline models generate synthetic gene expression data from metadata alone. You describe the biological conditions—tissue type, disease state, perturbations, cell type, etc.—and the model generates realistic expression profiles matching those conditions.

This is the most common use case: generating synthetic data for conditions where real data may be scarce or unavailable.

Available Models
----------------

- **gem-1-bulk**: Bulk RNA-seq baseline model
- **gem-1-sc**: Single-cell RNA-seq baseline model

.. code-block:: python

    import pysynthbio

Creating a Query
----------------

The structure of the query required by the API is specific to each model. Use ``get_example_query()`` to get a correctly structured example for your chosen model.

.. code-block:: python

    # Get the example query structure for a specific model
    example_query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]

    # Inspect the query structure
    print(example_query)

The query consists of:

1. **sampling_strategy**: The prediction mode that controls how expression data is generated:

   - **"sample generation"**: Generates realistic-looking synthetic data with measurement error (bulk only)
   - **"mean estimation"**: Provides stable mean estimates of expression levels (bulk and single-cell)

2. **inputs**: A list of biological conditions to generate data for

Each input contains ``metadata`` (describing the biological sample) and ``num_samples`` (how many samples to generate).

Making a Prediction
-------------------

Once your query is ready, send it to the API to generate gene expression data:

.. code-block:: python

    # Create a query for the bulk model
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]

    # Submit and get results
    result = pysynthbio.predict_query(query, model_id="gem-1-bulk")

The result is a dictionary containing two DataFrames: ``metadata`` and ``expression``.

Single-Cell Example
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Create a query for the single-cell model
    sc_query = pysynthbio.get_example_query(model_id="gem-1-sc")["example_query"]

    # Submit and get results
    sc_result = pysynthbio.predict_query(sc_query, model_id="gem-1-sc")

.. note::
   Single-cell models only support ``"mean estimation"`` mode.

Query Parameters
----------------

In addition to metadata, queries support several optional parameters that control the generation process.

sampling_strategy (str, required)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Controls the type of prediction the model generates. This parameter is required in all queries.

Available modes:

- **"sample generation"**: The model generates realistic-looking synthetic data that captures measurement error. This mode is useful when you want data that mimics real experimental measurements. **(Bulk only)**

- **"mean estimation"**: The model creates a distribution capturing biological heterogeneity consistent with the supplied metadata, then returns the mean of that distribution. This mode is useful when you want a stable estimate of expected expression levels. **(Bulk and single-cell)**

.. code-block:: python

    # Bulk query with sample generation
    bulk_query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]
    bulk_query["sampling_strategy"] = "sample generation"

    # Bulk query with mean estimation
    bulk_query_mean = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]
    bulk_query_mean["sampling_strategy"] = "mean estimation"

    # Single-cell query (must use mean estimation)
    sc_query = pysynthbio.get_example_query(model_id="gem-1-sc")["example_query"]
    sc_query["sampling_strategy"] = "mean estimation"  # Required for single-cell

total_count (int, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Library size used when converting predicted log CPM back to raw counts. Higher values scale counts up proportionally.

- Default: 10,000,000 for bulk; 10,000 for single-cell

.. code-block:: python

    # Create a query and add custom total_count
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]
    query["total_count"] = 5000000

deterministic_latents (bool, optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If ``True``, the model uses the mean of each latent distribution (``p(z|metadata)``) instead of sampling. This removes randomness from latent sampling and produces deterministic outputs for the same inputs.

- Default: ``False`` (sampling is enabled)

.. code-block:: python

    # Create a query and enable deterministic latents
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]
    query["deterministic_latents"] = True

seed (int, optional)
^^^^^^^^^^^^^^^^^^^^

Random seed for reproducibility when using stochastic sampling.

.. code-block:: python

    # Create a query with a specific seed
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]
    query["seed"] = 42

Combining Parameters
^^^^^^^^^^^^^^^^^^^^

You can combine multiple parameters in a single query:

.. code-block:: python

    # Create a query and add multiple parameters
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]
    query["total_count"] = 8000000
    query["deterministic_latents"] = True
    query["sampling_strategy"] = "mean estimation"

    results = pysynthbio.predict_query(query, model_id="gem-1-bulk")

Valid Metadata Keys
-------------------

The input metadata is a dictionary. Here is the full list of valid metadata keys:

Biological
^^^^^^^^^^

- ``age_years``
- ``cell_line_ontology_id``
- ``cell_type_ontology_id``
- ``developmental_stage``
- ``disease_ontology_id``
- ``ethnicity``
- ``genotype``
- ``race``
- ``sample_type`` ("cell line", "organoid", "other", "primary cells", "primary tissue", "xenograft")
- ``sex`` ("male", "female")
- ``tissue_ontology_id``

Perturbational
^^^^^^^^^^^^^^

- ``perturbation_dose`` (number and unit separated by a space, e.g., "10 um")
- ``perturbation_ontology_id``
- ``perturbation_time`` (number and unit separated by a space, e.g., "24 hours")
- ``perturbation_type`` ("coculture", "compound", "control", "crispr", "genetic", "infection", "other", "overexpression", "peptide or biologic", "shrna", "sirna")

Technical
^^^^^^^^^

- ``study`` (Bioproject ID)
- ``library_selection`` (e.g., "cDNA", "polyA", "Oligo-dT" - see `ENA documentation <https://ena-docs.readthedocs.io/en/latest/submit/reads/webin-cli.html#permitted-values-for-library-selection>`_)
- ``library_layout`` ("PAIRED", "SINGLE")
- ``platform`` ("illumina")

Valid Metadata Values
---------------------

The following are the valid values or expected formats for selected metadata keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metadata Field
     - Requirement / Example
   * - ``cell_line_ontology_id``
     - Requires a `Cellosaurus ID <https://www.cellosaurus.org/>`_
   * - ``cell_type_ontology_id``
     - Requires a `CL ID <https://www.ebi.ac.uk/ols4/ontologies/cl>`_
   * - ``disease_ontology_id``
     - Requires a `MONDO ID <https://www.ebi.ac.uk/ols4/ontologies/mondo>`_
   * - ``perturbation_ontology_id``
     - Must be a valid Ensembl gene ID (e.g., ``ENSG00000156127``), `ChEBI ID <https://www.ebi.ac.uk/chebi/>`_ (e.g., ``CHEBI:16681``), `ChEMBL ID <https://www.ebi.ac.uk/chembl/>`_ (e.g., ``CHEMBL1234567``), or `NCBI Taxonomy ID <https://www.ncbi.nlm.nih.gov/taxonomy>`_ (e.g., ``9606``)
   * - ``tissue_ontology_id``
     - Requires a `UBERON ID <https://www.ebi.ac.uk/ols4/ontologies/uberon>`_

We highly recommend using the `EMBL-EBI Ontology Lookup Service <https://www.ebi.ac.uk/ols4/>`_ to find valid IDs for your metadata.

Models have a limited acceptable range of metadata input values. If you provide a value that is not in the acceptable range, the API will return an error.

Modifying Query Inputs
----------------------

You can customize the query inputs to fit your specific research needs:

.. code-block:: python

    # Get a base query
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]

    # Adjust number of samples for the first input
    query["inputs"][0]["num_samples"] = 10

    # Add a new condition
    query["inputs"].append({
        "metadata": {
            "sex": "male",
            "sample_type": "primary tissue",
            "tissue_ontology_id": "UBERON:0002371"
        },
        "num_samples": 5
    })

Working with Results
--------------------

.. code-block:: python

    # Access metadata and expression matrices
    metadata = result["metadata"]
    expression = result["expression"]

    # Check dimensions
    print(expression.shape)

    # View metadata sample
    print(metadata.head())

You may want to process the data or save it for later use:

.. code-block:: python

    # Save results to files
    expression.to_csv("expression_matrix.csv")
    metadata.to_csv("sample_metadata.csv")

    # Or save as pickle for later use
    import pickle
    with open("synthesize_results.pkl", "wb") as f:
        pickle.dump(result, f)
