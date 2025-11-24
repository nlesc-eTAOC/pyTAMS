# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
#

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import pathlib
import sys
_rootpath = pathlib.Path(__file__).parents[1]
print('Project root path: {}'.format(_rootpath))
sys.path.insert(0, str(_rootpath))
sys.path.insert(0, str(_rootpath / 'pytams'))

# -- Project information -----------------------------------------------------

project = u"pyTAMS"
copyright = u"2023, Netherlands eScience Center"
author = u"Lucas Esclapez"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "0.0.6"
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# So that autodoc catch the word Attribute
napoleon_use_ivar = True
napolean_use_param = True
napoleon_google_docstring = True

# -- Use autoapi.extension to run sphinx-apidoc -------

autoapi_type = "python"
autoapi_dirs = ['../pytams']
autodoc_typehints = 'signature'
add_module_names = False
autoapi_template_dir = '_templates/autoapi'
autoapi_keep_files = True
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_add_toctree_entry = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = 'pyTAMS_documentation'

def setup(app):
    app.add_css_file('theme.css')

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# -- Options for Intersphinx

intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       # Commonly used libraries, uncomment when used in package
                       # 'numpy': ('http://docs.scipy.org/doc/numpy/', None),
                       # 'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
                       # 'scikit-learn': ('https://scikit-learn.org/stable/', None),
                       # 'matplotlib': ('https://matplotlib.org/stable/', None),
                       # 'pandas': ('http://pandas.pydata.org/docs/', None),
                       }
