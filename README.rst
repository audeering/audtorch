========
audtorch
========

Deep learning with PyTorch_ and audio.

audtorch_ collects audio related data sets, transforms and other useful
additions that are useful.

If you are interested in PyTorch_ and audio you should also check out the
efforts to integrate more audio directly into PyTorch_:

* `pytorch/audio`_
* `keunwoochoi/torchaudio-contrib`_

.. _PyTorch: https://pytorch.org
.. _audtorch: https://audtorch.readthedocs.io
.. _pytorch/audio: https://github.com/pytorch/audio
.. _keunwoochoi/torchaudio-contrib:
    https://github.com/keunwoochoi/torchaudio-contrib


Installation
============

audtorch_ is supported by Python 3.5 or higher. To install it run
(preferable in a `virtual environment`_):

.. code-block:: bash

    pip install audtorch

.. _audtorch: https://audtorch.readthedocs.io
.. _virtual environment: https://docs.python-guide.org/dev/virtualenvs


Usage
=====

audtorch_ provides classes for common audio data sets and a collection of
transforms that can handle numpy arrays.
To create fixed length speech samples augmented with white noise run:

.. code-block:: python

    >>> import sounddevice as sd
    >>> from audtorch import datasets, transforms
    >>> noise = datasets.WhiteNoise(duration=10240, sampling_rate=16000)
    >>> sample = transforms.Compose([transforms.RandomCrop(10240),
    ...                              transforms.RandomAdditiveMix(noise)])
    >>> data = datasets.LibriSpeech(root='~/LibriSpeech', sets='dev-clean',
    ...                             download=True, transform=sample)
    >>> signal, label = data[8]
    >>> sd.play(signal.transpose(), data.sampling_rate)

Besides data sets and transforms the package provides a few metrics, samplers,
and collate functons.

.. _audtorch: https://audtorch.readthedocs.io
