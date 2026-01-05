Getting Started
===============

``pysynthbio`` is a Python package that provides a convenient interface to the `Synthesize Bio <https://www.synthesize.bio/>`_ API, allowing users to generate realistic gene expression data based on specified biological conditions.
This package enables researchers to easily access AI-generated transcriptomic data for various modalities, including bulk RNA-seq and single-cell RNA-seq.

To generate datasets without code, use our `web platform <https://app.synthesize.bio/datasets/>`_.

Authentication
--------------

Get your API key
^^^^^^^^^^^^^^^^

Visit `the API Keys page <https://app.synthesize.bio/account/api-keys>`_ to generate an API key.
Click "+ Create API Key" then "Create Key", and copy your key.

There are multiple ways to set up your token:

Interactive Setup
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import pysynthbio

    # This opens a browser to the token creation page and prompts for input
    pysynthbio.set_synthesize_token(use_keyring=True)

If ``use_keyring=True``, the token persists across sessions; if ``use_keyring=False``, it is only set for the current session.
Keyring support is included by default in pysynthbio 2.2.1 and later.

Using Environment Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can set the ``SYNTHESIZE_API_KEY`` environment variable directly:

.. code-block:: bash

    export SYNTHESIZE_API_KEY=your_api_token_here  # macOS/Linux
    # Windows PowerShell:
    # $Env:SYNTHESIZE_API_KEY='your_api_token_here'

Non-Interactive Setup
^^^^^^^^^^^^^^^^^^^^^

For scripts running in non-interactive environments:

.. code-block:: python

    import pysynthbio

    # Supply token directly (avoid hardcoding secrets in source control)
    pysynthbio.set_synthesize_token(token="SECURE_SECRET_HERE")

Using the System Keyring
^^^^^^^^^^^^^^^^^^^^^^^^

If you've previously stored your token in the system keyring:

.. code-block:: python

    import pysynthbio

    # Attempt to load from keyring
    pysynthbio.load_synthesize_token_from_keyring()

Available Model Types
---------------------

Synthesize Bio provides several types of models for different use cases:

Baseline Models
^^^^^^^^^^^^^^^

Generate synthetic gene expression data from metadata alone. You describe the biological conditions (tissue type, disease state, perturbations, etc.) and the model generates realistic expression profiles.

- **gem-1-bulk**: Bulk RNA-seq baseline model
- **gem-1-sc**: Single-cell RNA-seq baseline model

See :doc:`baseline` for detailed usage.

Reference Conditioning Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generate expression data conditioned on a real reference sample. This allows you to "anchor" to an existing expression profile while applying perturbations or modifications.

- **gem-1-bulk_reference-conditioning**: Bulk RNA-seq reference conditioning model
- **gem-1-sc_reference-conditioning**: Single-cell RNA-seq reference conditioning model

See :doc:`reference-conditioning` for detailed usage.

Metadata Prediction Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Infer metadata from observed expression data. Given a gene expression profile, predict the likely biological characteristics (cell type, tissue, disease state, etc.).

- **gem-1-bulk_predict-metadata**: Bulk RNA-seq metadata prediction model
- **gem-1-sc_predict-metadata**: Single-cell RNA-seq metadata prediction model

See :doc:`metadata-prediction` for detailed usage.

Only baseline models are available to all users. You can check which models are available programmatically, see ``list_models()``. Contact us at support@synthesize.bio if you have any questions.

Listing Available Models
^^^^^^^^^^^^^^^^^^^^^^^^

You can check which models are available programmatically:

.. code-block:: python

    import pysynthbio

    # Check available models
    models = pysynthbio.list_models()
    print(models)

Quick Start
-----------

Here's a quick example using a baseline model:

.. code-block:: python

    import pysynthbio

    # Get an example query structure
    query = pysynthbio.get_example_query(model_id="gem-1-bulk")["example_query"]

    # Submit the query and get results
    result = pysynthbio.predict_query(query, model_id="gem-1-bulk")

    # Access the results
    metadata = result["metadata"]
    expression = result["expression"]

For more detailed examples and advanced usage, see the model-specific documentation linked above.

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

Troubleshooting
---------------

Keychain Access on Mac
^^^^^^^^^^^^^^^^^^^^^^

If you get this error on a Mac when using ``use_keyring=True``:

.. code-block:: none

   <stdin>:1: UserWarning: Failed to store token in keyring:
   Can't store password on keychain: (-25244, 'Unknown Error')

This occurs when your IDE or terminal does not have access to Keychain.
Go to System Preferences > Security & Privacy > Privacy > Full Disk Access and add the terminal or IDE you are working from (for example, Terminal, iTerm, VS Code, or PyCharm).
