from subprocess import check_output


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
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinxcontrib.katex',
    'sphinxcontrib.bibtex',
]

napoleon_use_ivar = True  # List of class attributes
autodoc_inherit_docstrings = False  # disable docstring inheritance

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

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
]

# HTML --------------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
}
html_title = title
