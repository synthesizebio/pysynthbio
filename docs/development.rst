.. _development-docs:

Documentation System
===================

This guide explains how to work with the PySynthBio documentation system, including building, testing, and deploying documentation.

Overview
--------

PySynthBio uses Sphinx to generate documentation from reStructuredText (RST) files. The documentation is automatically deployed to GitHub Pages when:

1. A new version is released to PyPI
2. A manual build is triggered via GitHub Actions

The documentation system supports multiple versions, allowing users to access documentation for any released version of PySynthBio.

Local Development
----------------

Building Documentation Locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The project includes several make commands to simplify working with documentation:

.. code-block:: bash

    # Build HTML documentation
    make docs

    # Build with warnings treated as errors (strict mode)
    make docs-strict

    # Start a live-reload server for documentation development
    make docs-live

    # Check for broken links
    make docs-linkcheck

    # Build a PDF version of the documentation
    make docs-pdf

    # Clean documentation build directory
    make docs-clean

    # Run all documentation tasks in sequence
    make docs-all

The built documentation will be available in the ``_build/html`` directory.

Adding New Documentation
~~~~~~~~~~~~~~~~~~~~~~~

To add new documentation:

1. Create a new ``.rst`` file in the appropriate directory
2. Add the file to the appropriate toctree in ``index.rst`` or a subject-specific index file
3. Build the documentation locally to ensure it renders correctly

For example, to add a new guide for a specific feature:

.. code-block:: rst

    .. _my-feature-guide:

    My Feature Guide
    ===============

    This guide explains how to use My Feature.

    ...

Versioned Documentation
----------------------

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~

The documentation system supports multiple versions with the following structure:

- ``/`` (root): Latest stable version
- ``/1.0.0/``: Specific version
- ``/2.0.0/``: Another specific version

A version selector in the top corner of each page allows users to switch between versions.

Updating Version Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Version information is automatically determined in several ways:

1. From GitHub release events when a new version is released
2. From tags generated during the PyPI release process
3. From manual version input when triggering the workflow

For local development, you can update the version in ``conf.py``:

.. code-block:: python

    release: str = "1.0.0"  # Change this to the version you're working on

GitHub Actions Workflows
-----------------------

The project includes several GitHub Actions workflows for documentation:

deploy-docs.yml
~~~~~~~~~~~~~~

This workflow deploys documentation to GitHub Pages. It is triggered:

1. Automatically after a successful PyPI release
2. Manually through the workflow dispatch with options for version and build settings

The workflow determines the appropriate version, builds the documentation, and deploys it to GitHub Pages with version management.

test-docs-build.yml
~~~~~~~~~~~~~~~~~~

This workflow tests documentation building without deploying to GitHub Pages. It is triggered:

1. On pull requests that modify documentation files
2. Manually through workflow dispatch

This workflow is helpful for testing documentation changes before merging them.

Best Practices
-------------

When working with documentation, follow these best practices:

1. **Test locally**: Always build and review documentation locally before pushing changes
2. **Use strict mode**: Use ``make docs-strict`` to catch and fix warnings early
3. **Check links**: Regular link checking with ``make docs-linkcheck`` prevents broken links
4. **Document as you code**: Update documentation when adding new features or changing existing ones
5. **Use cross-references**: Link related sections using Sphinx cross-references
6. **Include examples**: Provide practical examples for features
7. **Preview PRs**: Use the test-docs-build workflow to verify documentation changes in PRs

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

- **Missing dependencies**: Ensure you have all required packages with ``uv sync --all-extras --dev``
- **Sphinx errors**: Check for RST syntax errors with the ``docs-strict`` target
- **Broken links**: Use ``docs-linkcheck`` to find and fix broken links
- **Version mismatch**: Ensure version in ``conf.py`` matches the version you're working on

Getting Help
~~~~~~~~~~~

If you encounter issues with the documentation system:

1. Check existing GitHub issues for similar problems
2. Ask for help in the project's communication channels
3. Open a new issue with details of the problem 