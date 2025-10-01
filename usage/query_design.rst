Designing Queries for Models
============================
Choosing a Modality
^^^^^^^^^^^^^^^^^^^

``predict_query`` accepts a ``modality`` argument to select the data type to generate:

- ``bulk``: bulk RNA-seq (asynchronous under the hood, returned as DataFrames)
- ``czi``: single-cell RNA-seq (asynchronous under the hood, returned as DataFrames)

You do not need to specify any internal API slugs. The library maps modalities to the appropriate model endpoints automatically.

.. code-block:: python

    import pysynthbio

    q = pysynthbio.get_valid_query()

    # Bulk generation
    bulk = pysynthbio.predict_query(q, modality="bulk", as_counts=True)

    # Single-cell generation
    sc = pysynthbio.predict_query(q, modality="czi", as_counts=True)



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
