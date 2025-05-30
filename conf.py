# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project: str = "pysynthbio"
copyright: str = "2025, Candace Savonen, Alex David"
author: str = "Candace Savonen, Alex David"
release: str = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = []

templates_path: list[str] = []
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", ".venv", "venv"]

# Optional: If you have other file types or want to be explicit
source_suffix: dict[str, str] = {
    ".rst": "restructuredtext",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme: str = "alabaster"
html_static_path: list[str] = []
