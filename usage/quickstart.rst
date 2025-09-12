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

To see which modalities are supported by the current model, use ``get_valid_modalities``. This function returns a set of strings representing the allowed values for the ``modality`` key in your query.

.. code-block:: python

    supported_modalities = pysynthbio.get_valid_modalities()
    print(supported_modalities)
    # Output might look like: {'bulk', ...}

Generate Example Queries
------------------------

The structure of the query required by the API is fixed for the current supported model.
You can use ``get_valid_query`` to get a correctly structured example dictionary.

.. code-block:: python

    # Get the example query structure
    example_query = pysynthbio.get_valid_query()

Get Predictions
----------------

Use ``predict_query`` to send a query to the API and get expression predictions.
You'll typically use ``get_valid_query`` to help structure your request. This function also requires the API key.

.. code-block:: python

    # You can modify the example_query or create your own following the structure
    my_query = pysynthbio.get_valid_query() # Example: using the default valid query
    # Modify my_query as needed...

    results = pysynthbio.predict_query(
        query=my_query,
        as_counts=True # Get results as estimated counts (default); set to False for logCPM.
    )

    # Access results:
    metadata_df = results["metadata"]
    expression_df = results["expression"]

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
