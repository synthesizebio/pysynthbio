# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project: str = "pysynthbio"
copyright: str = "2025, Candace Savonen, Alex David"
author: str = "Candace Savonen, Alex David"
release: str = "3.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions: list[str] = []

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store", ".venv", "venv"]

# Optional: If you have other file types or want to be explicit
source_suffix: dict[str, str] = {
    ".rst": "restructuredtext",
}

# Inject a tiny script on every page to retarget the sidebar logo link
rst_prolog = """
.. raw:: html

   <script>(function(){function retarget(){try{var a=document.querySelector(
   '.sphinxsidebarwrapper p.logo a');if(a){a.href='https://www.synthesize.bio/';
   a.target='_blank';a.rel='noopener';}}catch(e){}} if(document.readyState===
   'loading'){document.addEventListener('DOMContentLoaded', retarget);}else{
   retarget();}})();</script>
    """

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme: str = "alabaster"
html_static_path: list[str] = ["_static"]
html_css_files: list[str] = ["custom.css"]
html_logo: str = "_static/logo.png"

# Ensure our about.html override is used
html_sidebars: dict[str, list[str]] = {
    "**": [
        "about.html",
        "navigation.html",
        "searchbox.html",
    ]
}
