.. _api-reference:

API Reference
============

This page provides details about the functions and classes available in the pysynthbio package.

Core Functions
-------------

get_valid_modalities()
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def get_valid_modalities() -> set[str]:
        """
        Get the valid modalities supported by the model.
        
        Returns:
            set[str]: A set of valid modality strings.
        """

Returns a set of strings representing the allowed values for the ``output_modality`` key in your query.

Example:

.. code-block:: python

    supported_modalities = pysynthbio.get_valid_modalities()
    print(supported_modalities)
    # Example output: {'bulk_rna-seq', 'lincs', 'sra', ...}

get_valid_query()
~~~~~~~~~~~~~~~

.. code-block:: python

    def get_valid_query() -> dict:
        """
        Get a valid example query structure.
        
        Returns:
            dict: A dictionary with the correct query structure.
        """

Returns a dictionary with the correct structure for making prediction queries. Use this as a template for your own queries.

Example:

.. code-block:: python

    example_query = pysynthbio.get_valid_query()
    print(example_query)

predict_query()
~~~~~~~~~~~~~

.. code-block:: python

    def predict_query(
        query: dict,
        as_counts: bool = True,
        api_key: Optional[str] = None
    ) -> dict[str, pd.DataFrame]:
        """
        Send a query to the API and get predictions.
        
        Args:
            query (dict): The query dictionary.
            as_counts (bool, optional): Whether to return results as counts. 
                                       Defaults to True.
            api_key (Optional[str], optional): API key. If None, will use 
                                              environment variable. Defaults to None.
        
        Returns:
            dict[str, pd.DataFrame]: Dictionary with 'metadata' and 'expression' DataFrames.
        """

Sends a query to the Synthesize Bio API and returns the results.

Parameters:

- ``query`` (dict): The query dictionary, typically created from ``get_valid_query()`` and modified.
- ``as_counts`` (bool, optional): Whether to return results as estimated counts (True) or logCPM (False). Defaults to True.
- ``api_key`` (str, optional): Your API key. If not provided, will look for the ``SYNTHESIZE_API_KEY`` environment variable.

Returns:

- dict[str, pd.DataFrame]: A dictionary with two keys:
  - ``metadata``: DataFrame containing metadata about the query results
  - ``expression``: DataFrame containing the expression values

Example:

.. code-block:: python

    my_query = pysynthbio.get_valid_query()
    # Modify my_query as needed...
    
    results = pysynthbio.predict_query(
        query=my_query,
        as_counts=True
    )
    
    metadata_df = results["metadata"]
    expression_df = results["expression"] 