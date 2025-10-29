Designing Queries for Models
============================
Choosing a Modality
^^^^^^^^^^^^^^^^^^^

The modality (data type to generate) is specified in the query dictionary using ``get_valid_query()``:

- ``bulk``: bulk RNA-seq (asynchronous under the hood, returned as DataFrames)
- ``single-cell``: single-cell RNA-seq (asynchronous under the hood, returned as DataFrames)

You do not need to specify any internal API slugs. The library maps modalities to the appropriate model endpoints automatically.

.. code-block:: python

    import pysynthbio

    # Create a bulk query
    bulk_query = pysynthbio.get_valid_query(modality="bulk")
    bulk = pysynthbio.predict_query(bulk_query, as_counts=True)

    # Create a single-cell query
    sc_query = pysynthbio.get_valid_query(modality="single-cell")
    sc = pysynthbio.predict_query(sc_query, as_counts=True)



Valid Metadata Keys
^^^^^^^^^^^^^^^^^^^

The structure of the query required by the API is fixed for the current supported model.
You can use ``get_valid_query`` to get a correctly structured example dictionary.

.. code-block:: python

    import pysynthbio
    # Get the example query structure
    example_query = pysynthbio.get_valid_query()

This is the full list of valid metadata keys:

- ``age_years``
- ``cell_line_ontology_id``
- ``cell_type_ontology_id``
- ``developmental_stage``
- ``disease_ontology_id``
- ``ethnicity``
- ``genotype``
- ``perturbation_dose``
- ``perturbation_ontology_id``
- ``perturbation_time``
- ``perturbation_type``
- ``race``
- ``sample_type``
- ``sex``
- ``tissue_ontology_id``
- ``study``
- ``library_selection``
- ``library_layout``
- ``platform``

Valid Metadata Values
^^^^^^^^^^^^^^^^^^^^^

The following are the valid values or expected formats for selected metadata keys:

- ``cell_line_ontology_id``: Requires a `Cellosaurus ID <https://www.cellosaurus.org/>`_.
- ``cell_type_ontology_id``: Requires a `CL ID <https://www.ebi.ac.uk/ols/ontologies/cl>`_.
- ``disease_ontology_id``: Requires a `MONDO ID <https://www.ebi.ac.uk/ols/ontologies/mondo>`_.
- ``perturbation_ontology_id``: Must be a valid Ensembl gene ID (e.g., ``ENSG00000156127``), `ChEBI ID <https://www.ebi.ac.uk/chebi/>`_ (e.g., ``CHEBI:16681``), `ChEMBL ID <https://www.ebi.ac.uk/chembl/>`_ (e.g., ``CHEMBL1234567``), or `NCBI Taxonomy ID <https://www.ncbi.nlm.nih.gov/taxonomy>`_ (e.g., ``9606``).
- ``tissue_ontology_id``: Requires a `UBERON ID <https://www.ebi.ac.uk/ols/ontologies/uberon>`_.

We highly recommend using the `EMBL-EBI Ontology Lookup Service <https://www.ebi.ac.uk/ols4/>`_ to find valid IDs for your metadata.

Models have a limited acceptable range of metadata input values.
If you provide a value that is not in the acceptable range, the API will return an error.

Query Parameters
^^^^^^^^^^^^^^^^

In addition to metadata, queries support several optional parameters that control the generation process:

**mode** (str)
    Controls the type of prediction the model generates. This parameter is required in all queries.

    Available modes:

    - **"sample generation"**: The model works identically to the mean estimation approach, except that the final gene expression distribution is also sampled to generate realistic-looking synthetic data that captures the error associated with measurements. This mode is useful when you want data that mimics real experimental measurements.

    - **"mean estimation"**: The model creates a distribution capturing the biological heterogeneity consistent with the supplied metadata. This distribution is then sampled to predict a gene expression distribution that captures measurement error. The mean of that distribution serves as the prediction. This mode is useful when you want a stable estimate of expected expression levels.

    .. note::
       **Single-cell queries only support "mean estimation" mode.** Bulk queries support both modes.

    .. code-block:: python

        import pysynthbio

        # Bulk query with sample generation
        bulk_query = pysynthbio.get_valid_query(modality="bulk")
        bulk_query["mode"] = "sample generation"  # Default for bulk

        # Bulk query with mean estimation
        bulk_query_mean = pysynthbio.get_valid_query(modality="bulk")
        bulk_query_mean["mode"] = "mean estimation"

        # Single-cell query (must use mean estimation)
        sc_query = pysynthbio.get_valid_query(modality="single-cell")
        sc_query["mode"] = "mean estimation"  # Required for single-cell

**total_count** (int)
    Library size used when converting predicted log CPM back to raw counts. Higher values scale counts up proportionally.

    .. code-block:: python

        import pysynthbio

        # Create a query and add custom total_count
        query = pysynthbio.get_valid_query(modality="bulk")
        query["total_count"] = 5000000

**deterministic_latents** (bool)
    If true, the model uses the mean of each latent distribution (``p(z|metadata)`` or ``q(z|x)``) instead of sampling. This removes randomness from latent sampling and produces deterministic outputs for the same inputs.

    - Default: false (sampling is enabled)

    .. code-block:: python

        import pysynthbio

        # Create a query and enable deterministic latents
        query = pysynthbio.get_valid_query(modality="bulk")
        query["deterministic_latents"] = True

You can combine multiple parameters in a single query:

.. code-block:: python

    import pysynthbio

    # Create a query and add multiple parameters
    query = pysynthbio.get_valid_query(modality="bulk")
    query["total_count"] = 8000000
    query["deterministic_latents"] = True

    results = pysynthbio.predict_query(query)
