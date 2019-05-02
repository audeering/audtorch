import torch
from torch.nn.utils.rnn import pad_sequence


class Collation(object):
    r"""Abstract interface for collation classes.

    All other collation classes should subclass it. All subclasses should
    override ``__call__``, that executes the actual collate function.

    """

    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        raise NotImplementedError("This is an abstract interface for "
                                  "modularizing collate functions."
                                  "Please use one of its subclasses.")


class Seq2Seq(Collation):
    r"""Pads mini-batches to longest contained sequence for seq2seq-purposes.

    This class pads features and targets to the largest sequence in the batch.
    Before padding, length information are extracted from them.

    Note:
        The tensors can be sorted in descending order of features' lengths
        by enabling :attr:`sort_sequences`. Thereby the requirements of
        :py:func:`torch.nn.utils.rnn.pack_padded_sequence`
        are anticipated, which is used by recurrent layers.

    * :attr:`sequence_dimensions` holds dimension of sequence in features
      and targets
    * :attr:`batch_first` controls output shape of features and targets
    * :attr:`pad_values` controls values to pad features (targets) with
    * :attr:`sort_sequences` controls if sequences are sorted in
      descending order of `features`' lengths

    Args:
        sequence_dimensions (list of ints): indices representing dimension of
            sequence in feature and target tensors.
            Position `0` represents sequence dimension of `features`,
            position `1` represents sequence dimension of `targets`.
            Negative indexing is permitted
        batch_first (bool or None, optional): determines output shape of
            collate function. If `None`, original shape of
            `features` and `targets` is kept with dimension of `batch size`
            prepended. See Shape for more information.
            Default: `None`
        pad_values (list, optional): values to pad shorter sequences with.
            Position `0` represents value of `features`,
            position `1` represents value of `targets`. Default: `[0, 0]`
        sort_sequences (bool, optional): option whether to sort sequences
            in descending order of `features`' lengths. Default: `True`

    Shape:
        - Input: :math:`(*, S, *)`, where :math:`*` can be any number
          of further dimensions except :math:`N` which is the batch size,
          and where :math:`S` is the sequence dimension.
        - Output:

          - `features`:

            - :math:`(N, *, S, *)` if :attr:`batch_first` is `None`,
              i.e. the original input shape with :math:`N` prepended
              which is the batch size
            - :math:`(N, S, *, *)` if :attr:`batch_first` is `True`
            - :math:`(S, N, *, *)` if :attr:`batch_first` is `False`

          - `feats_lengths`: :math:`(N,)`

          - `targets`: analogous to `features`

          - `tgt_lengths`: analogous to `feats_lengths`

    Example:
        >>> # data format: FS = (feature dimension, sequence dimension)
        >>> batch = [[torch.zeros(161, 108), torch.zeros(10)],
        ...          [torch.zeros(161, 223), torch.zeros(12)]]
        >>> collate_fn = Seq2Seq([-1, -1], batch_first=None)
        >>> features = collate_fn(batch)[0]
        >>> list(features.shape)
        [2, 161, 223]

    """

    def __init__(self, sequence_dimensions, batch_first=None,
                 pad_values=[0, 0], sort_sequences=True):

        self.sequence_dimensions = sequence_dimensions
        self.batch_first = batch_first
        self.pad_values = pad_values
        self.sort_sequences = sort_sequences

    def __call__(self, batch):
        r"""Collate and pad sequences of mini-batch.

        The output tensor is augmented by the dimension of `batch_size`.

        Args:
            batch (list of tuples): contains all samples of a batch.
                Each sample is represented by a tuple (`features`, `targets`)
                which is returned by data set's __getitem__ method

        Returns:
            torch.tensors: `features`, `feature lengths`, `targets`
                and `target lengths` in data format according to
                :attr:`batch_first`.

        """
        features = [torch.as_tensor(sample[0]) for sample in batch]
        features, feats_lengths, sorted_indices = _collate_sequences(
            features, self.sequence_dimensions[0], self.pad_values[0],
            self.batch_first, self.sort_sequences, [])

        targets = [torch.as_tensor(sample[1]) for sample in batch]
        targets, tgt_lengths, _ = _collate_sequences(
            targets, self.sequence_dimensions[1], self.pad_values[1],
            self.batch_first, self.sort_sequences, sorted_indices)

        return features, feats_lengths, targets, tgt_lengths


def _collate_sequences(sequences, sequence_dimension, pad_value,
                       batch_first, sort_sequences=True, sorted_indices=[]):
    r"""Collate and pad sequences.

    Args:
        sequences (list of torch.tensors): contains all samples of a batch
        sequence_dimension (int): index representing dimension of sequence
            in tensors
        batch_first (bool or None): determines output shape of tensors
        pad_value (float): value to pad shorter sequences with.
        sort_sequences (bool, optional): option whether to sort `sequences`.
            Default: `True`
        sorted_indices (list of ints, optional): indices to sort sequences
            and their lengths in descending order with. Default: `[]`

    Returns:
        tuple:

        * torch.Tensor: data of sequences in format
          according to :attr:`batch_first`, list of sorted indices
        * torch.IntTensor: lengths of sequences
        * list of int: indices of sequences sorted in descending order
          of their lengths

    """
    # handle negative indexing
    sequence_dimension = len(sequences[0].shape) + sequence_dimension \
        if sequence_dimension < 0 else sequence_dimension

    # input to `pad_sequence` requires shape `(S, *)`
    # swap sequence-dimension to the front
    sequences = [t.transpose(0, sequence_dimension) for t in sequences]

    # extract lengths (first dimension of permuted tensor)
    lengths = [t.shape[0] for t in sequences]

    if sort_sequences:
        # sort sequences and lengths in descending order of `lengths`
        if not sorted_indices:
            sorted_indices = sorted(
                range(len(lengths)), key=lambda i: lengths[i], reverse=True)

        sequences = [sequences[idx] for idx in sorted_indices]
        lengths = [lengths[idx] for idx in sorted_indices]

    # pad sequences
    sequences = pad_sequence(
        sequences=sequences,
        batch_first=batch_first if batch_first is not None else True,
        padding_value=pad_value)

    if batch_first is None:  # recover input data format
        sequences = sequences.transpose(1, sequence_dimension + 1)

    else:  # recover order of "*"-dimensions
        permuted = list(range(len(sequences.shape)))
        if sequence_dimension >= 2:
            permuted.insert(
                2, permuted.pop(sequence_dimension + 1))
        sequences = sequences.permute(permuted)

    lengths = torch.IntTensor(lengths)

    return sequences, lengths, sorted_indices
