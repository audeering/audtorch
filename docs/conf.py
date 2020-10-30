import os
import sys
from subprocess import check_output


# Import ------------------------------------------------------------------

# Relative source code path. Avoids `import audtorch`, which need package to be
# installed first.
sys.path.insert(0, os.path.abspath('..'))

# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    'audiofile',
    'librosa',
    'numpy',
    'pandas',
    'resampy',
    'torch',
    'scipy',
    'tqdm',
    'tabulate',
]


# Project -----------------------------------------------------------------

project = 'audtorch'
copyright = '2019 audEERING GmbH'
author = ('Andreas Triantafyllopoulos, '
          'Stephan Huber, '
          'Johannes Wagner, '
          'Hagen Wierstorf')
# The x.y.z version read from tags
try:
    version = check_output(['git', 'describe', '--tags', '--always'])
    version = version.decode().strip()
except Exception:
    version = '<unknown>'
title = '{} Documentation'.format(project)


# General -----------------------------------------------------------------

master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = None
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx_copybutton',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.katex',
    'nbsphinx',
]

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = False  # disable docstring inheritance

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

copybutton_prompt_text = r'>>> |\.\.\. |$ '
copybutton_prompt_is_regexp = True

nbsphinx_execute = 'never'

linkcheck_ignore = [
    'https://doi.org/',  # has timeouts from time to time
]

# HTML --------------------------------------------------------------------

html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'footer_links': False,
    'logo_only': False,
}
html_context = {
    'display_github': True,
}

html_title = title
