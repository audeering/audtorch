import numpy as np
from torch.utils.data import Sampler


class BucketSampler(Sampler):
    r"""Creates batches from ordered data sets.

    This sampler iterates over the data sets of `concat_dataset`
    and samples sequentially from them.
    Samples of each batch deliberately originate solely from the same data set.
    Only when the current data set is exhausted, the next data set
    is sampled from. In other words,
    samples from different buckets are never mixed.

    In each epoch `num_batches` batches of size `batch_sizes`
    are extracted from each data set.
    If the requested number of batches cannot be extracted from a data set,
    only its available batches are queued.
    By default, the data sets (and thus their batches) are iterated over
    in increasing order of their data set id.

    Note:
        The information in :attr:`batch_sizes` and :attr:`num_batches`
        refer to :attr:`datasets` at the same index
        independently of :attr:`permuted_order`.

    Simple Use Case: "Train on data with increasing sequence length"

    ======================= ==================================
    bucket_id:              [0,     1,      2,     ...  end  ]
    batch_sizes:            [32,    16,     8,     ...  2    ]
    num_batches:            [None,  None,   None,  ...  None ]
    ======================= ==================================

    Result:
    "Extract all batches (`None`) from all data sets,
    all of different batch size, and queue them
    in increasing order of their data set id"

    * :attr:`batch_sizes` controls batch size for each data set
    * :attr:`num_batches` controls number of batches to extract
      from each data set
    * :attr:`permuted_order` controls if order in which data sets are iterated
      over is permuted or in which specific order iteration is permuted
    * :attr:`shuffle_each_bucket` controls if each data set is shuffled
    * :attr:`drop_last` controls whether to drop last samples of a bucket
      which cannot form a mini-batch

    Args:
        concat_dataset (torch.utils.data.ConcatDataset): ordered concatenated
            data set
        batch_sizes (list): batch sizes per data set. Permissible values are
            unsigned integers
        num_batches (list or None, optional): number of batches per data set.
            Permissible values are non-negative integers and None.
            If None, then as many batches are extracted as data set provides.
            Default: `None`
        permuted_order (bool or list, optional): option whether to permute the
            order of data set ids in which the respective data set's batches
            are queued. If True (False), data set ids are (not) shuffled.
            Besides, a customized list of permuted data set ids can be
            specified. Default: `False`
        shuffle_each_bucket (bool, optional): option whether to shuffle samples
            in each data set. Recommended to set to True. Default: `True`
        drop_last (bool, optional): controls whether the last samples of a
            bucket which cannot form a mini-batch should be dropped.
            Default: `False`

    Example:
        >>> import torch
        >>> from torch.utils.data import (TensorDataset, ConcatDataset)
        >>> from audtorch.datasets.utils import defined_split
        >>> data = TensorDataset(torch.randn(100))
        >>> lengths = np.random.randint(0, 890, (100,))
        >>> split_func = buckets_of_even_size(lengths, num_buckets=3)
        >>> subsets = defined_split(data, split_func)
        >>> concat_dataset = ConcatDataset(subsets)
        >>> batch_sampler = BucketSampler(concat_dataset, 3 * [16])

    """

    def __init__(self, concat_dataset, batch_sizes, num_batches=None,
                 permuted_order=False, shuffle_each_bucket=True,
                 drop_last=False):
        self.datasets = _stack_concatenated_datasets(concat_dataset)
        self.batch_sizes = batch_sizes
        self.num_batches = num_batches
        self.permuted_order = permuted_order
        self.shuffle_each_bucket = shuffle_each_bucket
        self.drop_last = drop_last
        self.dataset_ids = list(range(len(self.datasets)))

        if isinstance(permuted_order, list):
            assert sorted(self.permuted_order) == self.dataset_ids, \
                '`permuted_order` not consistent with number of data sets.'
            self.dataset_ids = permuted_order

        assert all(
            [self.batch_sizes[i] > 0 and isinstance(self.batch_sizes[i], int)
             for i in self.dataset_ids]
        ), 'Only positive integers permitted for `num_batches`.'

        if self.num_batches is not None:
            assert all(
                [self.num_batches[i] >= 0 and isinstance(
                    self.num_batches[i], int) for i in self.dataset_ids
                 if self.num_batches[i] is not None]), \
                "Only non-negative integers or " \
                "`None` permitted for `num_batches`."

        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

    def __iter__(self):
        r"""Iterates sequentially over data sets and forms batches

        """
        all_batches = []
        batch = []

        if self.permuted_order is True:
            self.dataset_ids = list(np.random.permutation(self.dataset_ids))

        # iterate over data sets ordered by data set id
        for dset_id in self.dataset_ids:

            dataset = self.datasets[dset_id]
            num_batch = 0

            if self.shuffle_each_bucket:  # random samples from data set
                dataset = list(np.random.permutation(dataset))

            for sample in dataset:  # iterate over samples in data set
                if self.num_batches is not None and \
                        self.num_batches[dset_id] is not None:
                    if num_batch == self.num_batches[dset_id]:
                        break
                batch.append(sample)
                if len(batch) == self.batch_sizes[dset_id]:
                    all_batches.append(batch)
                    num_batch += 1
                    batch = []

            # yield full batch and also \
            # handle last samples of bucket which cannot form entire batch
            if len(batch) > 0 and not self.drop_last:
                all_batches.append(batch)
            batch = []

        return iter(all_batches)

    def __len__(self):

        sampler_size = 0
        for dset_id in self.dataset_ids:

            dataset = self.datasets[dset_id]
            bs = self.batch_sizes[dset_id]
            requested_batches = None if self.num_batches is None \
                else self.num_batches[dset_id]

            if self.drop_last:
                fitted_batches = len(dataset) // bs
            else:
                fitted_batches = (len(dataset) + bs - 1) // bs

            if requested_batches is None:
                sampler_size += fitted_batches
            else:
                sampler_size += fitted_batches if \
                    fitted_batches < requested_batches \
                    else requested_batches

        return sampler_size


def buckets_by_boundaries(key_values, bucket_boundaries):
    r"""Split samples into buckets based on key values using bucket boundaries.

    Note:
        A sample is sorted into bucket :math:`i` if for their key value
        holds:

            :math:`b_{i-1} <= \text{key value} < b_i`,
            where :math:`b_i` is `bucket boundary` at index :math:`i`

    Args:
        key_values (list): contains key values, e.g. sequence lengths
        bucket_boundaries (list): contains boundaries of buckets in
            ascending order. The list should neither contain a lower or
            upper boundary, e.g. not numpy.iinfo.min or numpy.iinfo.max.

    Returns:
        func: Key function to use for splitting: \
        :math:`f(\text{item}) = \text{bucket\_id}`

    Example:
        >>> lengths = [288, 258, 156, 99, 47, 13]
        >>> boundaries = [80, 150]
        >>> split_func = buckets_by_boundaries(lengths, boundaries)
        >>> [split_func(i) for i in range(len(lengths))]
        [2, 2, 2, 1, 0, 0]

    """
    assert bucket_boundaries == sorted(bucket_boundaries), \
        "Iterable `bucket_boundaries` not given in ascending order."

    assert len(bucket_boundaries) == \
        len(np.unique(bucket_boundaries)), \
        "Iterable `bucket_boundaries` contains duplicate(s)."

    num_buckets = len(bucket_boundaries) + 1

    def key_func(item):
        key_val = key_values[item]
        for bucket_id in range(num_buckets - 1):
            if key_val < bucket_boundaries[bucket_id]:
                return bucket_id
        return num_buckets - 1

    return key_func


def buckets_of_even_size(key_values, num_buckets, reverse=False):
    r"""Split samples into buckets of even size based on key values.

    The samples are sorted with either increasing (or decreasing) key value.
    If number of samples cannot be distributed evenly to buckets,
    the first buckets are filled up with one remainder each.

    Args:
        key_values (list): contains key values, e.g. sequence lengths
        num_buckets (int): number of buckets to form. Permitted are
            positive integers
        reverse (bool, optional): if True, then sort in descending order.
            Default: `False`

    Returns:
        func: Key function to use for splitting: \
        :math:`f(\text{item}) = \text{bucket\_id}`

    Example:
        >>> lengths = [288, 258, 156, 47, 112, 99, 13]
        >>> num_buckets = 4
        >>> split_func = buckets_of_even_size(lengths, num_buckets)
        >>> [split_func(i) for i in range(len(lengths))]
        [3, 2, 2, 0, 1, 1, 0]

    """
    # make sure that bucket size is larger than 0
    assert (len(key_values) >= num_buckets), \
        "Not enough `key_values` for `num_buckets` in order to" \
        " form even buckets."

    assert (num_buckets > 0) and isinstance(num_buckets, int), \
        "Specified value for `num_buckets` not a positive integer."

    bucket_size, remainder = divmod(len(key_values), num_buckets)
    sorted_indices = sorted(range(len(key_values)),
                            key=lambda i: key_values[i],
                            reverse=reverse)

    bucket_membership = len(key_values) * [0]
    sample_count, bucket_id, r = 0, 0, remainder

    for i in sorted_indices:
        bucket_membership[i] = bucket_id
        sample_count += 1
        if sample_count == bucket_size:  # bucket full
            # distribute one remainder per bucket
            if r > 0 and bucket_id + r == remainder:
                r -= 1
                sample_count -= 1
            else:
                sample_count = 0
                bucket_id += 1

    def key_func(item):
        return bucket_membership[item]

    return key_func


def _stack_concatenated_datasets(concat_dataset):
    r"""Extract and stack indices of different data sets from `concat_dataset`.

    Each data set is represented by a complete range of indices
    starting from 0...len(data set) for the first data set.
    The ranges of the following data sets build on
    the range of the corresponding previous data set so that
    the indices of the lists are cumulative::

        datasets = [[0 ... len(data_1) - 1],
                    [len(data_1) ... len(data_1) - 1], ... ]

    Args:
        concat_dataset (ConcatDataset): data set to sample from

    Returns:
        list: list of lists of data set indices

    Example:
        >>> import torch
        >>> from torch.utils.data import (TensorDataset, ConcatDataset)
        >>> data_1 = TensorDataset(torch.Tensor([0, 1, 2, 3]))
        >>> data_2 = TensorDataset(torch.Tensor([0, 1, 2]))
        >>> concat_dataset = ConcatDataset((data_1, data_2))
        >>> concat_dataset.cumulative_sizes
        [4, 7]
        >>> _stack_concatenated_datasets(concat_dataset)
        [[0, 1, 2, 3], [4, 5, 6]]

    """
    datasets = []
    start_idx = 0
    for upper_edge in concat_dataset.cumulative_sizes:
        datasets += [list(range(start_idx, upper_edge))]
        start_idx = upper_edge

    return datasets
