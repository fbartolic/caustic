# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'caustic'
copyright = '2019, Fran Bartolic'
author = 'Fran Bartolic'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'nbsphinx',
]


#autodoc_mock_imports = ["numpy", "scipy", "astropy", "pymc3", "theano"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "astropy": ("http://docs.astropy.org/en/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']

nbsphinx_prompt_width = 0


# -- Extension configuration -------------------------------------------------
master_doc = 'index' # fix contents.rst not found error

source_suffix = ['.rst', '.ipynb']

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base='docs') %}

.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. nbinfo::

        This page was generated from `{{ docname }}`__.
        Interactive online version:
        :raw-html:`<a href="https://mybinder.org/v2/gh/fbartolic/caustic/{{ env.config.release }}?filepath={{ docname }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>`

    __ https://github.com/fbartolic/caustic
        {{ env.config.release }}/{{ docname }}

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
    """
