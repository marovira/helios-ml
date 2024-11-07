# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Helios"
copyright = "2024, Mauricio A. Rovira Galvez"  # noqa: A001
author = "Mauricio A. Rovira Galvez"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]
autosummary_generate = True
templates_path = ["_templates"]
exclude_patterns = []  # type: ignore[var-annotated]

autoapi_dirs = ["../../src"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {
    "sidebar_hide_name": True,
}
html_static_path = ["_static"]
html_logo = "../../data/logo/logo-transparent.png"
