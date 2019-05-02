audtorch.collate
================

Collate functions manipulate and merge a list
of samples to form a mini-batch, see :py:class:`torch.utils.data.DataLoader`.
An example use case is batching sequences of variable-length,
which requires padding each sample to the maximum length in the batch.

.. automodule:: audtorch.collate

Collation
---------

.. autoclass:: Collation
    :members:

Seq2Seq
-------

.. autoclass:: Seq2Seq
    :members:
