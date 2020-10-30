audtorch.transforms.functional
==============================

The goal of the transform functionals is to provide functions that work
independent on the dimensions of the input signal and can be used easily to
create the actual transforms.

.. Note::

    All of the transforms work currently only with :py:obj:`numpy.array` as
    inputs, not :py:obj:`torch.Tensor`.

.. automodule:: audtorch.transforms.functional

crop
----

.. autofunction:: crop

pad
---

.. autofunction:: pad

replicate
---------

.. autofunction:: replicate

downmix
-------

.. autofunction:: downmix

upmix
-----

.. autofunction:: upmix

additive_mix
------------

.. autofunction:: additive_mix

mask
----

.. autofunction:: mask

normalize
---------

.. autofunction:: normalize

standardize
-----------

.. autofunction:: standardize

stft
----

.. autofunction:: stft

istft
-----

.. autofunction:: istft
