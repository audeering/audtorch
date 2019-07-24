import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from audtorch import (datasets, samplers, transforms)


xfail = pytest.mark.xfail
filterwarnings = pytest.mark.filterwarnings


# --- datasets/noise.py ---
@pytest.mark.parametrize('duration', [0.01, 0.1, 1])
@pytest.mark.parametrize('sampling_rate', [8000])
@pytest.mark.parametrize('mean', [0, 1])
@pytest.mark.parametrize('stdev', [1, 0.5])
def test_whitenoise(duration, sampling_rate, mean, stdev):
    dataset = datasets.WhiteNoise(duration=duration,
                                  sampling_rate=sampling_rate,
                                  mean=mean,
                                  stdev=stdev)
    noise, label = next(iter(dataset))
    samples = int(np.ceil(duration * sampling_rate))
    assert noise.shape == (1, samples)
    assert label == 'white noise'
    assert -1 <= np.max(np.abs(noise)) <= 1
    assert len(dataset) == 1


# --- datasets/utils.py ---
crop = transforms.RandomCrop(8192)
resamp1 = transforms.Resample(48000, 44100)
resamp2 = transforms.Resample(44100, 16000)
t1 = transforms.Compose([crop, resamp1])
t2 = transforms.Compose([crop, resamp1, resamp2])
t3 = transforms.Compose([resamp1, crop, resamp2])
d0 = datasets.WhiteNoise(duration=0.5, sampling_rate=48000, transform=crop)
d1 = datasets.WhiteNoise(duration=0.5, sampling_rate=48000, transform=t1)
d2 = datasets.WhiteNoise(duration=0.5, sampling_rate=48000, transform=t2)
d3 = datasets.WhiteNoise(duration=0.5, sampling_rate=48000, transform=t3)
df_empty = pd.DataFrame()
df_a = pd.DataFrame(data=[0], columns=['a'])
df_ab = pd.DataFrame(data=[('0', 1)], columns=['a', 'b'])


@pytest.mark.parametrize('list_of_datasets', [
    (d2, d3),
    pytest.param([d0, d1], marks=xfail(raises=ValueError))
])
def test_audioconcatdataset(list_of_datasets):
    datasets.AudioConcatDataset(list_of_datasets)


@pytest.mark.parametrize(('input,expected_output,expected_sampling_rate,'
                          'expected_warning'), [
    ('', np.array([[]]), None, 'File does not exist: '),
])
def test_load(input, expected_output, expected_sampling_rate,
              expected_warning):
    with pytest.warns(UserWarning, match=expected_warning):
        output, sampling_rate = datasets.load(input)
    assert np.array_equal(output, expected_output)
    assert sampling_rate == expected_sampling_rate


@pytest.mark.parametrize('transform', [t1, t2, t3])
def test_sampling_rate_after_transform(transform):
    expected_sampling_rate = transform.transforms[-1].output_sampling_rate
    dataset = datasets.WhiteNoise(duration=0.5,
                                  sampling_rate=48000,
                                  transform=transform)
    sampling_rate = datasets.sampling_rate_after_transform(dataset)
    assert sampling_rate == expected_sampling_rate
    assert dataset.sampling_rate == expected_sampling_rate


@pytest.mark.parametrize('list_of_datasets', [
    [d0, d0, d0],
    [d1, d1],
    [d2, d2],
    pytest.param([1], marks=xfail(raises=RuntimeError)),
    pytest.param([d0, d1], marks=xfail(raises=ValueError)),
    pytest.param([d1, d2], marks=xfail(raises=ValueError)),
])
def test_ensure_same_sampling_rate(list_of_datasets):
    datasets.ensure_same_sampling_rate(list_of_datasets)


@pytest.mark.parametrize('df,labels', [
    pytest.param(df_empty, 'a', marks=xfail(raises=RuntimeError)),
    (df_a, 'a'),
    pytest.param(df_a, 'b', marks=xfail(raises=RuntimeError)),
    (df_ab, ['a', 'b']),
    (df_ab, 'a'),
    pytest.param(df_ab, 'c', marks=xfail(raises=RuntimeError)),
])
def test_ensure_df_columns_contain(df, labels):
    datasets.ensure_df_columns_contain(df, labels)


@pytest.mark.parametrize('df', [
    pytest.param(df_empty, marks=xfail(raises=RuntimeError)),
    df_a,
])
def test_ensure_df_not_empty(df):
    datasets.ensure_df_not_empty(df)


@pytest.mark.parametrize('df,expected_files,expected_labels', [
    (df_ab, ['./0'], [1]),
    (None, [], []),
    pytest.param(df_empty, [], [], marks=xfail(raises=RuntimeError)),
])
def test_files_and_labels_from_df(df, expected_files, expected_labels):
    files, labels = datasets.files_and_labels_from_df(df,
                                                      root='.',
                                                      column_filename='a',
                                                      column_labels='b')
    expected_files = [datasets.safe_path(f) for f in expected_files]
    assert files == expected_files
    assert labels == expected_labels


@pytest.mark.parametrize('key_values,split_func,kwargs,expected', [
    (list(range(5)), samplers.buckets_of_even_size, 2, [3, 2]),
    (list(range(10)), samplers.buckets_by_boundaries, [0, 3, 8], [3, 5, 2])
])
def test_defined_split(key_values, split_func, kwargs, expected):
    data = TensorDataset(torch.arange(len(key_values)))
    split_func = split_func(torch.Tensor(key_values), kwargs)
    subsets = datasets.defined_split(data, split_func)

    assert expected == [len(subset) for subset in subsets]
