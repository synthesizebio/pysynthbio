Quickstart
==========

``pysynthbio`` is a Python package that provides a convenient interface to the `Synthesize Bio <https://www.synthesize.bio/>`_ API, allowing users to generate realistic gene expression data based on specified biological conditions.
This package enables researchers to easily access AI-generated transcriptomic data for various modalities, including bulk RNA-seq and single-cell RNA-seq.

To generate datasets without code, use our `web platform <https://app.synthesize.bio/datasets/>`_.
This guide will help you get started with ``pysynthbio`` quickly.

Get your API key
----------------

Visit `<https://app.synthesize.bio/account/api-keys>`_ to generate an API key.
Click "+ Create API Key" then "Create Key", and copy your key.

There are multiple ways to set up your token:

Interactive Setup
-----------------

.. code-block:: python

    import pysynthbio

    # This opens a browser to the token creation page and prompts for input
    pysynthbio.set_synthesize_token(use_keyring=True)

If ``use_keyring=True``, the token persists across sessions; if ``use_keyring=False``, it is only set for the current session.
Keyring support is included by default in pysynthbio 2.2.1 and later.

Using Environment Variables
---------------------------

You can set the ``SYNTHESIZE_API_KEY`` environment variable directly:

.. code-block:: bash

    export SYNTHESIZE_API_KEY=your_api_token_here  # macOS/Linux
    # Windows PowerShell:
    # $Env:SYNTHESIZE_API_KEY='your_api_token_here'


Non-Interactive Setup
---------------------

For scripts running in non-interactive environments:

.. code-block:: python

    import pysynthbio

    # Supply token directly (avoid hardcoding secrets in source control)
    pysynthbio.set_synthesize_token(token="SECURE_SECRET_HERE")

Using the System Keyring
------------------------
If you've previously stored your token in the system keyring:

.. code-block:: python

    import pysynthbio

    # Attempt to load from keyring
    pysynthbio.load_synthesize_token_from_keyring()


Import the package
-------------------

.. code-block:: python

    import pysynthbio


Discover Valid Modalities
-------------------------

The API supports multiple output modalities. Use ``get_valid_modalities`` to see what is available. The two primary modalities are ``'bulk'`` and ``'single-cell'``.

.. code-block:: python

    supported_modalities = pysynthbio.get_valid_modalities()
    print(supported_modalities)
    # Example: {'bulk', 'single-cell'}

Generate Example Queries
------------------------

The structure of the query required by the API is fixed for the current supported model.
You can use ``get_valid_query`` to get a correctly structured example dictionary.

.. code-block:: python

    # Get the example query structure
    example_query = pysynthbio.get_valid_query()

Get Predictions
----------------

Use ``predict_query`` to send a query to the API and get expression predictions. You choose the data generation type with the ``modality`` parameter:

- ``modality='bulk'`` generates bulk RNA-seq.
- ``modality='single-cell'`` generates single-cell RNA-seq.

The function handles authentication, request submission, and result retrieval. For single-cell, the API runs asynchronously; ``predict_query`` automatically polls the job until it's ready and then downloads the results for you.

.. code-block:: python

    # Start from a valid base query
    my_query = pysynthbio.get_valid_query()

    # Example 1: Generate bulk counts (default)
    bulk_results = pysynthbio.predict_query(
        query=my_query,
        modality="bulk",
        as_counts=True,  # counts; set False for logCPM
    )
    bulk_metadata = bulk_results["metadata"]
    bulk_expression = bulk_results["expression"]

    # Example 2: Generate single-cell counts (async under-the-hood)
    sc_results = pysynthbio.predict_query(
        query=my_query,
        modality="single-cell",
        as_counts=True,
    )
    sc_metadata = sc_results["metadata"]
    sc_expression = sc_results["expression"]

Advanced Options
----------------

Query Parameters
^^^^^^^^^^^^^^^^

The query dictionary supports several optional parameters to control the generation process:

.. code-block:: python

    # Create a query with custom parameters
    my_query = pysynthbio.get_valid_query(
        modality="bulk",
        total_count=8000000,        # Custom library size
        deterministic_latents=True  # Deterministic output
    )
    
    results = pysynthbio.predict_query(query=my_query)

Available query parameters:

- **total_count** (int): Library size for converting log CPM to counts. Defaults: 10,000,000 (bulk), 10,000 (single-cell)
- **deterministic_latents** (bool): If True, uses mean of latent distributions instead of sampling for reproducible results

You can also manually add these to any query dictionary:

.. code-block:: python

    my_query = pysynthbio.get_valid_query()
    my_query["total_count"] = 5000000
    my_query["deterministic_latents"] = False
    
    results = pysynthbio.predict_query(query=my_query)

API Function Parameters
^^^^^^^^^^^^^^^^^^^^^^^

``predict_query`` also exposes a few optional parameters you can use to tune API behavior:

- ``poll_interval_seconds``: Seconds between status checks for async jobs (default 2).
- ``poll_timeout_seconds``: Maximum time to wait before raising a timeout error (default 15 minutes).
- ``api_base_url``: Override the API host (useful for local testing or staging). Defaults to production.

This covers the basic workflow: understanding the required query structure and making predictions.

Security Notes
--------------

- The API token provides full access to your Synthesize Bio account.
- When using ``use_keyring=True``, your token is stored securely in your system's credential manager.
- For production environments, consider using environment variables or secrets management tools.

Cleanup
-------

When you're done using the API, you can clear the token from your environment:

.. code-block:: python

    # Clear from current session
    pysynthbio.clear_synthesize_token()

    # Clear from both session and system keyring
    pysynthbio.clear_synthesize_token(remove_from_keyring=True)



Rate Limits
-----------

Free usage of Synthesize Bio is limited.
If you exceed this limit, you may receive an error from the API explaining the limit.
If you need to generate more samples, please contact us at support@synthesize.bio for more information.

Troubleshooting Note
--------------------

If you get this error on a Mac when using ``use_keyring=True``:

.. code-block:: none

   <stdin>:1: UserWarning: Failed to store token in keyring:
   Can't store password on keychain: (-25244, 'Unknown Error')

This occurs when your IDE or terminal does not have access to Keychain.
Go to System Preferences > Security & Privacy > Privacy > Full Disk Access and add the terminal or IDE you are working from (for example, Terminal, iTerm, VS Code, or PyCharm).
