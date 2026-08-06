# Configuration file for the Sphinx documentation builder.
import importlib_metadata

# -- Project information -----------------------------------------------------
project = 'sollertia-video-tracking'
copyright = '2026, Sun (NeuroAI) lab'
author = 'Ivan Kondratyev'
# Extracts the project version from the metadata .toml file.
release = importlib_metadata.version("sollertia-video-tracking")

# -- General configuration ---------------------------------------------------
# The sphinx_click entry must load before sphinx_autodoc_typehints. With the two swapped, the click directive fails to
# import slvt_cli with TypeError: 'module' object is not callable, and the build still exits zero, logging that as a
# non-fatal ERROR while the Command-Line Interface heading renders with no commands beneath it. Confirmed with
# sphinx-click 6.2.0, sphinx-autodoc-typehints 3.13.2, and Sphinx 9.1.0. Re-check when any of the three is upgraded.
extensions = [
    'sphinx.ext.autodoc',        # To build documentation from python source code docstrings.
    'sphinx.ext.napoleon',       # To read google-style docstrings (works with autodoc module).
    'sphinx_click',              # Must load before sphinx_autodoc_typehints. See the note above the list.
    'sphinx_autodoc_typehints',  # To parse typehints into documentation
]

templates_path = ['_templates']
exclude_patterns = []

# Google-style docstring parsing configuration for napoleon extension
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Additional sphinx-typehints configuration
always_document_param_types = False
typehints_document_rtype = True
typehints_use_rtype = True
typehints_defaults = 'comma'
simplify_optional_unions = True
typehints_formatter = None
typehints_use_signature = False
typehints_use_signature_return = False

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
