Installation
============

Prerequisites
-------------

To start using pysynthbio, first you need to have an account with synthesize.bio.
`Go here to create one <https://app.synthesize.bio/profile>`_.

Standard Installation
---------------------

Then on your machine you can install using pip:

.. code-block:: bash

    pip install pysynthbio

To ensure it installed correctly you can run:

.. code-block:: bash

    pip show pysynthbio

Installing from a GitHub Release
--------------------------------

Alternatively, you can install a specific version directly from its GitHub Release page. This is useful for testing pre-releases or specific tagged versions.

1. Go to the `Releases page <https://github.com/synthesizebio/pysynthbio/releases>`_ of the repository.
2. Find the release tag you want to install (e.g., ``v0.1.0``).
3. Expand the "Assets" section for that release.
4. Download the ``.whl`` (wheel) file or the ``.tar.gz`` (source distribution) file. The wheel file is generally preferred if available for your platform.
5. Install the downloaded file using pip:

.. code-block:: bash

    # Example using a downloaded wheel file
    pip install /path/to/pysynthbio-0.1.0-py3-none-any.whl

    # Example using a downloaded source distribution
    pip install /path/to/pysynthbio-0.1.0.tar.gz

Development Installation
------------------------

For development purposes, you can install directly from the repository:

.. code-block:: bash

    git clone https://github.com/synthesizebio/pysynthbio.git
    cd pysynthbio
    pip install -e . 