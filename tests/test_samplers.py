import pytest
import torch
import numpy as np
from numpy.random import randint
import random
from bisect import bisect_right
from torch.utils.data import (TensorDataset, ConcatDataset)

from audtorch.samplers import (
    buckets_of_even_size, buckets_by_boundaries, BucketSampler)
from audtorch.datasets.utils import defined_split


# data sets
data_size = 1000
num_feats = 160
max_length = 300
max_feature = 100
lengths = torch.randint(0, max_length, (data_size,))
inputs = torch.randint(0, max_feature, (data_size, num_feats, max_length))
data = TensorDataset(inputs)

# function params
num_buckets = randint(1, 10)
batch_sizes = num_buckets * [randint(1, np.iinfo(np.int8).max)]
num_batches = num_buckets * [randint(0, np.iinfo(np.int8).max)]
bucket_boundaries = [b + num for num, b in enumerate(
    sorted(list(random.sample(range(max_length - num_buckets),
                              (num_buckets - 1)))))]


@pytest.mark.parametrize("key_values", [lengths])
@pytest.mark.parametrize("num_buckets", [num_buckets])
@pytest.mark.parametrize("reverse", [True, False])
def test_buckets_of_even_size(key_values, num_buckets, reverse):

    expected_bucket_size, remainders = divmod(data_size, num_buckets)
    expected_bucket_dist = remainders * [expected_bucket_size + 1]
    expected_bucket_dist += (num_buckets - remainders) * [expected_bucket_size]

    buckets = buckets_of_even_size(key_values=key_values,
                                   num_buckets=num_buckets,
                                   reverse=reverse)

    expected_bucket_ids = set(range(num_buckets))
    bucket_ids = [buckets(i) for i in range(data_size)]
    key_values = [key_values[i] for i in range(data_size)]
    bucket_dist = [bucket_ids.count(bucket_id)
                   for bucket_id in range(num_buckets)]

    # do bucket ids only range from 0 to num_buckets-1?
    assert expected_bucket_ids == set(bucket_ids)

    sort_indices = sorted(range(data_size), key=lambda idx: key_values[idx],
                          reverse=reverse)
    sorted_buckets = [bucket_ids[idx] for idx in sort_indices]
    diff_buckets = np.diff(sorted_buckets)

    # sorted with monotonously increasing/decreasing length of key values?
    assert all(diff >= 0 for diff in diff_buckets)

    # are buckets evenly distributed (except for remainders)?
    assert expected_bucket_dist == bucket_dist


@pytest.mark.parametrize("key_values", [lengths])
@pytest.mark.parametrize("bucket_boundaries", [bucket_boundaries])
def test_buckets_by_boundaries(key_values, bucket_boundaries):

    buckets = buckets_by_boundaries(key_values=lengths,
                                    bucket_boundaries=bucket_boundaries)

    num_buckets = len(bucket_boundaries) + 1
    expected_bucket_ids = list(range(num_buckets))

    data_size = key_values.shape[0]
    bucket_ids = [buckets(idx) for idx in range(data_size)]
    key_values = [key_values[idx] for idx in range(data_size)]

    # do bucket ids only range from 0 to num_buckets-1?
    # missing ids only allowed if corresponding bucket empty
    missing_buckets = list(set(expected_bucket_ids) - set(bucket_ids))
    for bucket_id in missing_buckets:
        if bucket_id == 0:
            assert not any([v < bucket_boundaries[bucket_id]
                            for v in key_values])
        elif bucket_id == expected_bucket_ids[-1]:
            assert not any([v > bucket_boundaries[(bucket_id - 1)]
                            for v in key_values])
        else:
            assert not any([v in range(bucket_boundaries[(bucket_id - 1)],
                                       bucket_boundaries[bucket_id])
                            for v in key_values])

    sort_indices = sorted(range(data_size), key=lambda i: key_values[i])
    sorted_buckets = [bucket_ids[i] for i in sort_indices]
    diff_buckets = np.diff(sorted_buckets)

    # sorted with monotonously increasing/decreasing length of key values?
    assert all(diff >= 0 for diff in diff_buckets)


@pytest.mark.parametrize("data", [data])
@pytest.mark.parametrize(
    "key_func",
    [buckets_of_even_size(lengths, num_buckets, reverse=False),
     buckets_by_boundaries(lengths, bucket_boundaries)])
@pytest.mark.parametrize("expected_num_datasets", [num_buckets])
@pytest.mark.parametrize("batch_sizes", [batch_sizes])
@pytest.mark.parametrize("expected_num_batches", [num_batches])
@pytest.mark.parametrize("permuted_order", [False, random.shuffle(
    list(range(num_buckets)))])
@pytest.mark.parametrize("shuffle_each_bucket", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_bucket_sampler(data, key_func, expected_num_datasets,
                        batch_sizes, expected_num_batches, permuted_order,
                        shuffle_each_bucket, drop_last):

    subsets = defined_split(data, key_func)
    concat_dataset = ConcatDataset(subsets)

    batch_sampler = BucketSampler(
        concat_dataset=concat_dataset,
        batch_sizes=batch_sizes,
        num_batches=expected_num_batches,
        permuted_order=permuted_order,
        shuffle_each_bucket=shuffle_each_bucket,
        drop_last=drop_last)

    num_datasets = len(batch_sampler.datasets)
    expected_dataset_ids = list(range(num_datasets))
    if isinstance(permuted_order, list) and permuted_order:
        expected_dataset_ids = permuted_order

    # assert data sets via batch sampler
    batch_indices = list(iter(batch_sampler))
    epoch_batch_sizes = [len(batch) for batch in batch_indices]
    dataset_sizes = [len(d) for d in batch_sampler.datasets]

    expected_epoch_batch_sizes = []
    expected_dset_ids = []

    for i in expected_dataset_ids:

        skip = False

        if batch_sizes[i] == 0 or expected_num_batches[i] == 0:
            continue

        fitted_batches = dataset_sizes[i] // batch_sizes[i]
        num_batches = fitted_batches
        if expected_num_batches[i] <= fitted_batches:
            num_batches = expected_num_batches[i]
            skip = True

        add_batch_sizes = [batch_sizes[i]] * num_batches

        if not drop_last and not skip:
            remainder = dataset_sizes[i] % batch_sizes[i]
            if remainder != 0:
                add_batch_sizes += [remainder]

        expected_epoch_batch_sizes += add_batch_sizes

        if len(add_batch_sizes) > 0:
            expected_dset_ids += [i]

    # are batch sizes as expected?
    assert expected_epoch_batch_sizes == epoch_batch_sizes

    unique_batch_ids = []
    for batch in batch_indices:
        ids = list(set([bisect_right(
            concat_dataset.cumulative_sizes, idx) for idx in batch]))
        unique_batch_ids += [ids]

    # are all samples of each batch from identical data set?
    assert all(list(map(lambda l: len(l) == 1, unique_batch_ids)))

    # flatten list
    unique_batch_ids = [i for ids in unique_batch_ids for i in ids]

    dataset_ids = []
    prev_id = None

    for i in unique_batch_ids:
        current_id = i
        if prev_id is None:
            dataset_ids += [current_id]
        else:
            if not current_id == prev_id:
                dataset_ids += [current_id]
        prev_id = current_id

    # are batches drawn from data sets in desired order?
    assert expected_dset_ids == dataset_ids
