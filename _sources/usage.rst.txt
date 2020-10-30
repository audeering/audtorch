Usage
=====

:mod:`audtorch` automates the data iteration process for deep neural
network training using PyTorch_. It provides a set of feature extraction
transforms that can be implemented on-the-fly on the CPU.

The following example creates a data set of speech samples that are cut to a
fixed length of 10240 samples. In addition they are augmented on the fly during
data loading by a transform that adds samples from another data set:

.. code-block:: python

    >>> import sounddevice as sd
    >>> from audtorch import datasets, transforms
    >>> noise = datasets.WhiteNoise(duration=10240, sampling_rate=16000)
    >>> augment = transforms.Compose([transforms.RandomCrop(10240),
    ...                               transforms.RandomAdditiveMix(noise)])
    >>> data = datasets.LibriSpeech(root='~/LibriSpeech', sets='dev-clean',
    ...                             download=True, transform=augment)
    >>> signal, label = data[8]
    >>> sd.play(signal.transpose(), data.sampling_rate)

Besides data sets and transforms the package provides standard evaluation
metrics, samplers, and necessary collate functions for training deep neural
networks for audio tasks.

.. _PyTorch: https://pytorch.org
