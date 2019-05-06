import pytest
import torch
import numpy as np
import random
from bisect import bisect_right
from torch.utils.data import TensorDataset, ConcatDataset

from audtorch.samplers import (
    buckets_of_even_size, buckets_by_boundaries, BucketSampler)
from audtorch.datasets.utils import defined_split

# dataset parameters
data_size = 1000
num_feats = 160
max_seq_len = 3000
feats_range = (0, 1000)
seq_len_range = (0, max_seq_len)

# variables and datasets
lens = torch.randint(*seq_len_range, (data_size,))
inputs = torch.randint(*feats_range, (data_size, num_feats, max_seq_len))
dataset = TensorDataset(inputs)

# function params
num_datasets = 10
batch_sizes = [np.random.randint(1, np.iinfo(np.int8).max)
               for _ in range(num_datasets)]
num_batches = [np.random.randint(0, np.iinfo(np.int8).max)
               for _ in range(num_datasets)]
bucket_boundaries = [b + num for num, b in enumerate(
    sorted(list(random.sample(range(max_seq_len - num_datasets),
                              (num_datasets - 1)))))]


@pytest.mark.parametrize("keys", [lens])
@pytest.mark.parametrize("num_buckets", [num_datasets])
@pytest.mark.parametrize("reverse", [True, False])
def test_buckets_of_even_size(keys, num_buckets, reverse):

    data_size = keys.shape[0]
    expected_bucket_size = data_size // num_buckets
    expected_bucket_dist = (num_buckets - 1) * [expected_bucket_size]
    expected_bucket_dist += \
        [expected_bucket_size + data_size % num_buckets]

    key_func = buckets_of_even_size(
        key_values=keys,
        num_buckets=num_buckets,
        reverse=reverse)

    expected_bucket_ids = set(range(num_buckets))
    bucket_ids = [key_func(i) for i in range(data_size)]
    key_values = [keys[i] for i in range(data_size)]
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

    # are buckets evenly distributed (except last)?
    assert expected_bucket_dist == bucket_dist


@pytest.mark.parametrize("keys", [lens])
@pytest.mark.parametrize("bucket_boundaries", [bucket_boundaries])
def test_buckets_by_boundaries(keys, bucket_boundaries):

    key_func = buckets_by_boundaries(
        key_values=lens,
        bucket_boundaries=bucket_boundaries)

    num_buckets = len(bucket_boundaries) + 1
    expected_bucket_ids = list(range(num_buckets))

    data_size = keys.shape[0]
    bucket_ids = [key_func(idx) for idx in range(data_size)]
    key_values = [keys[idx] for idx in range(data_size)]

    # do bucket ids only range from 0 to num_buckets-1?
    # missing ids only allowed if corresponding bucket empty
    missing_buckets = list(set(expected_bucket_ids) - set(bucket_ids))
    for bucket_id in missing_buckets:
        if bucket_id == 0:
            assert not any([key < bucket_boundaries[bucket_id]
                            for key in key_values])
        elif bucket_id == expected_bucket_ids[-1]:
            assert not any([key > bucket_boundaries[(bucket_id - 1)]
                            for key in key_values])
        else:
            assert not any([key in
                            range(bucket_boundaries[(bucket_id - 1)],
                                  bucket_boundaries[bucket_id])
                            for key in key_values])

    sort_indices = sorted(range(data_size), key=lambda idx: key_values[idx])
    sorted_buckets = [bucket_ids[idx] for idx in sort_indices]
    diff_buckets = np.diff(sorted_buckets)

    # sorted with monotonously increasing/decreasing length of key values?
    assert all(diff >= 0 for diff in diff_buckets)


@pytest.mark.parametrize("dataset", [dataset])
@pytest.mark.parametrize(
    "key_func",
    [buckets_of_even_size(lens, num_datasets, False),
     buckets_by_boundaries(lens, bucket_boundaries)])
@pytest.mark.parametrize("expected_num_datasets", [num_datasets])
@pytest.mark.parametrize("batch_sizes", [batch_sizes])
@pytest.mark.parametrize("expected_num_batches", [num_batches])
@pytest.mark.parametrize("permuted_order", [False, random.shuffle(
    list(range(num_datasets)))])
@pytest.mark.parametrize("shuffle_each_bucket", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_bucket_sampler(dataset, key_func, expected_num_datasets,
                        batch_sizes, expected_num_batches, permuted_order,
                        shuffle_each_bucket, drop_last):

    datasets = defined_split(dataset, key_func)
    concat_dataset = ConcatDataset(datasets)

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

        if batch_sizes[i] is 0 or expected_num_batches[i] is 0:
            continue

        fitted_batches = dataset_sizes[i] // batch_sizes[i]
        num_batches = fitted_batches
        if expected_num_batches[i] <= fitted_batches:
            num_batches = expected_num_batches[i]
            skip = True

        add_batch_sizes = [batch_sizes[i]] * num_batches

        if not drop_last and not skip:
            remainder = dataset_sizes[i] % batch_sizes[i]
            if remainder is not 0:
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
