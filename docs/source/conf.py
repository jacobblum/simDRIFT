import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))
sys.path.insert(0, os.path.abspath('../../src/setup'))
sys.path.insert(0, os.path.abspath('../../src/physics'))
sys.path.insert(0, os.path.abspath('../../src/configs'))
sys.path.insert(0, os.path.abspath('../../src/data'))
sys.path.insert(0, os.path.abspath('../../src/gui'))
sys.path.insert(0, os.path.abspath('../../src/jp'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SimDRIFT'
copyright = '2023, Jacob S. Blum and Kainen L. Utt'
author = 'Jacob S. Blum and Kainen L. Utt, Ph.D.'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
'sphinx.ext.intersphinx',
'sphinx.ext.mathjax',
'sphinx.ext.coverage',
'sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = []
mathjax3_config = {
  'loader': {'load': ['[tex]/physics','[tex]/upgreek']},
  'tex': {'packages': {'[+]': ['physics','upgreek']}}
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for LaTeX output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/latex.html

latex_engine = 'xelatex'
latex_elements = {
    'preamble': r'''
\usepackage[titles]{tocloft}
\usepackage{amsmath,amsthm,amsfonts,amssymb}
\usepackage{physics}
\cftsetpnumwidth {1.25cm}\cftsetrmarg{1.5cm}
\setlength{\cftchapnumwidth}{0.75cm}
\setlength{\cftsecindent}{\cftchapnumwidth}
\setlength{\cftsecnumwidth}{1.25cm}
''',
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'printindex': r'\footnotesize\raggedright\printindex',
}
latex_show_urls = 'footnote'
