# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "py-s-d-TDGL"
copyright = "2026, Nikhil Kodali; based on pyTDGL"
author = "Nikhil Kodali and pyTDGL contributors"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "enum_tools.autoenum",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinxarg.ext",
    "sphinxcontrib.bibtex",
    "sphinx.ext.autosectionlabel",
    # https://github.com/spatialaudio/nbsphinx/issues/24
    # https://github.com/spatialaudio/nbsphinx/issues/687
    "IPython.sphinxext.ipython_console_highlighting",
]

bibtex_bibfiles = [
    "references.bib",
]

# Make sure the target is unique
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 1

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "./images/logo-transparent-small.png"
html_theme_options = {
    "logo_only": False,
    "display_version": False,
}

napoleon_use_param = True

napoleon_google_docstring = True

autodoc_member_order = "bysource"

autodoc_typehints = "description"

nbsphinx_execute = "never"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'retina', 'png'}",
    "import matplotlib.pyplot as plt",
    "plt.rcParams['figure.figsize'] = (6, 4)",
    "plt.rcParams['font.size'] = 10",
]

math_eqref_format = "Eq. {number}"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

if os.environ.get("SPHINX_OFFLINE"):
    intersphinx_mapping = {}
