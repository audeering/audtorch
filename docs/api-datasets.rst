audtorch.datasets
=================

Audio data sets.

.. automodule:: audtorch.datasets


Common
------

This section contains a mix of generic data sets that are useful for a wide
variety of cases and can be used as base classes for other data sets.

AudioDataset
~~~~~~~~~~~~

.. autoclass:: AudioDataset
    :members:

PandasDataset
~~~~~~~~~~~~~

.. autoclass:: PandasDataset
    :members:

CsvDataset
~~~~~~~~~~

.. autoclass:: CsvDataset
    :members:

AudioConcatDataset
~~~~~~~~~~~~~~~~~~

.. autoclass:: AudioConcatDataset
    :members:


Speech
------

MozillaCommonVoice
~~~~~~~~~~~~~~~~~~

.. autoclass:: MozillaCommonVoice
    :members:


LibriSpeech
~~~~~~~~~~~

.. autoclass:: LibriSpeech
    :members:


Scene Analysis
--------------

This section contains data sets that can be primarily used for the analysis of
acoustic scenes.

AudioSet
~~~~~~~~

.. autoclass:: AudioSet
    :members:


Noise
-----

This section contains data sets that are primarily used as noise sources.

WhiteNoise
~~~~~~~~~~

.. autoclass:: WhiteNoise
    :members:


utils
-----

Utility functions for handling audio data sets.

load
~~~~

.. autofunction:: load

download_url
~~~~~~~~~~~~

.. autofunction:: download_url

download_url_list
~~~~~~~~~~~~~~~~~

.. autofunction:: download_url_list

extract_archive
~~~~~~~~~~~~~~~

.. autofunction:: extract_archive

sampling_rate_after_transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: sampling_rate_after_transform

ensure_same_sampling_rate
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ensure_same_sampling_rate

ensure_df_columns_contain
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ensure_df_columns_contain

ensure_df_not_empty
~~~~~~~~~~~~~~~~~~~

.. autofunction:: ensure_df_not_empty

files_and_labels_from_df
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: files_and_labels_from_df
