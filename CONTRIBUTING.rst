Contributing
============

Everyone is invited to contribute to this project. Feel free to create a
`pull request`_.
If you find errors, omissions, inconsistencies or other things that need
improvement, please create an issue_.

.. _issue: https://github.com/audeering/audtorch/issues/new/
.. _pull request: https://github.com/audeering/audtorch/compare/


Development Installation
------------------------

Instead of pip-installing the latest release from PyPI_, you should get the
newest development version from Github_::

   git clone https://github.com/audeering/audtorch/
   cd audtorch
   # Create virtual environment, e.g.
   # virtualenv --python=python3 _env
   # source _env/bin/activate
   python setup.py develop

.. _PyPI: https://pypi.org/project/audtorch/
.. _Github: https://github.com/audeering/audtorch/

This way, your installation always stays up-to-date, even if you pull new
changes from the Github_ repository.

If you prefer, you can also replace the last command with::

   pip install -r requirements.txt


Pull requests
-------------

When creating a new pull request, please conside the following points:

* Focus on a single topic as it is easier to review short pull requests
* Ensure your code is readable and `PEP 8`_ compatible
* Provide a test for proposed new functionality
* Add a docstring, see the `Writing Documentation` remarks below
* Choose a `meaningful commit messages`_

.. _PEP 8: https://www.python.org/dev/peps/pep-0008/
.. _meaningful commit messages: https://chris.beams.io/posts/git-commit/


Writing Documentation
---------------------

The API documentation of :mod:`audtorch` is build automatically from the
docstrings_ of its classes and functions.

docstrings_ are written in reStructuredText_ as indicated by the ``r`` at
its beginning and they are written using the `Google docstring convention`_
with the following additions:

* Start argument description in lower case and end the last sentence without a
  punctation.
* If the argument is optional, its default value has to be indicated.
* Description of attributes start as well in lower case and stop without
  punctuation.
* Attributes that can influence the behavior of the class should be described by
  the word ``controls``.
* Attributes that are supposed to be read only and provide only information
  should be described by the word ``holds``.
* Have a special section for class attributes.
* Python variables should be set in single back tics in the description of the
  docstring, e.g. ```True```. Only for some explicit statements like a list
  of variables it might be look better to write them as code, e.g.
  ```'mean'```.

The important part of the docstrings_ is the first line which holds a short
summary of the functionality, that should not be longer than one line, written
in imperative, and stops with a point. It is also considered good practice to
include an usage example.

reStructuredText_ allows for easy inclusion of math in LaTeX syntax that will
be dynamically rendered in the browser.

After you are happy with your docstring, you have to include it into the main
documentation under the ``docs/`` folder in the appropriate api file. E.g.
``energy()`` is part of the ``utils`` module and the corresponding file in the
documentation would be ``docs/api-utils.rst``, where it is included.

.. _docstrings: https://www.python.org/dev/peps/pep-0257/
.. _reStructuredText:
    http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _Google docstring convention:
    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html


Building Documentation
----------------------

If you make changes to the documentation, you can re-create the HTML pages
using Sphinx_.
You can install it and a few other necessary packages with::

    pip install -r doc/requirements.txt

To create the HTML pages, use::

    sphinx-build docs/ build/sphinx/html/ -b html

The generated files will be available in the directory ``build/sphinx/html/``.

It is also possible to automatically check if all links are still valid::

    sphinx-build docs/ build/sphinx/html/ -b linkcheck

.. _Sphinx: http://sphinx-doc.org/


Running Tests
-------------

You'll need pytest_ and a few dependencies for that.
It can be installed with::

   pip install -r tests/requirements.txt

To execute the tests, simply run::

   pytest

.. _pytest: https://pytest.org/


Creating a New Release
----------------------

New releases are made using the following steps:

#. Update ``CHANGELOG.rst``
#. Commit those changes as "Release X.Y.Z"
#. Create an (annotated) tag with ``git tag -a X.Y.Z``
#. Push the commit and the tag to Github
