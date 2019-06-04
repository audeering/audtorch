import pytest
import numpy as np
import librosa

import audtorch.transforms.functional as F


xfail = pytest.mark.xfail

a11 = np.array([1, 2, 3, 4], dtype=float)
a12 = np.array([5, 6, 7, 8], dtype=float)
a21 = np.array([9, 10, 11, 12], dtype=float)
a22 = np.array([13, 14, 15, 16], dtype=float)
ones = np.ones(4)
zeros = np.zeros(4)
A = np.array([[a11, a12], [a21, a22]])  # Tensor of shape (2, 2, 4)

# Ratio in dB to add two inputs to yield 1.5 magnitude
_half_ratio = -10 * np.log10(0.5 ** 2)


def _mean(input, axis):
    """Return mean along axis and preserve number of dimensions."""
    return np.expand_dims(np.mean(input, axis=axis), axis=axis)


def _pad(vector, padding, value):
    """Add padding to a vector using np.pad."""
    return np.pad(vector, padding, 'constant', constant_values=value)


@pytest.mark.parametrize('input,idx,axis,expected_output', [
    (A, (0, 2), -1, np.array([[a11[:2], a12[:2]], [a21[:2], a22[:2]]])),
    (A, (1, 2), -2, np.array([[a12], [a22]])),
    (A, (0, 1), 0, np.array([[a11, a12]])),
    (A, -1, -2, np.array([[a12], [a22]])),
    (A, 0, -2, np.array([[a11], [a21]])),
    (a11, (1, 2), -1, a11[1:2]),
])
def test_crop(input, idx, axis, expected_output):
    output = F.crop(input, idx, axis=axis)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize('input,padding,value,axis,expected_output', [
    (A, 1, 0, -1, np.array([[_pad(a11, 1, 0), _pad(a12, 1, 0)],
                            [_pad(a21, 1, 0), _pad(a22, 1, 0)]])),
    (A, (0, 1), 1, 1, np.array([[a11, a12, ones], [a21, a22, ones]])),
    (A, (1, 0), 0, 0, np.array([[zeros, zeros], [a11, a12], [a21, a22]])),
    (a11, 1, 1, -1, _pad(a11, 1, 1)),
])
def test_pad(input, padding, value, axis, expected_output):
    output = F.pad(input, padding, value=value, axis=axis)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize('input,repetitions,axis', [
    (A, 2, -1),
    (A, 3, 0),
    (A, 4, 1),
    (a11, 1, 0)])
def test_replicate(input, repetitions, axis):
    expected_output = np.concatenate(tuple([input] * repetitions), axis)
    output = F.replicate(input, repetitions, axis=axis)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize('input,channels,method,axis,expected_output', [
    (A, 2, 'mean', -2, A),
    (A, 1, 'mean', -2, _mean(A, axis=-2)),
    (A, 1, 'crop', -2, np.array([[a11], [a21]])),
    (A, 0, 'mean', -2, np.empty((2, 0, 4))),  # empty array with correct shape
    (A, 0, 'crop', -2, np.empty((2, 0, 4))),
    (A, 1, 'mean', 0, _mean(A, axis=0)),
    (a11, 1, 'crop', -1, np.array([a11[0]])),
    (a11, 1, 'crop', -2, np.array(a11)),
])
def test_downmix(input, channels, method, axis, expected_output):
    output = F.downmix(input, channels, method=method, axis=axis)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize('input,channels,method,axis,expected_output', [
    (A, 2, 'mean', -2, A),
    (A, 3, 'mean', -2, np.hstack((A, _mean(A, axis=-2)))),
    (A, 3, 'zero', -2, np.hstack((A, [[zeros], [zeros]]))),
    (A, 3, 'repeat', -2, np.hstack((A, A[:, -1, None, :]))),
    (A, 3, 'mean', 0, np.vstack((A, _mean(A, axis=0)))),
    (a11, 2, 'repeat', -2, np.array([a11, a11])),
])
def test_upmix(input, channels, method, axis, expected_output):
    output = F.upmix(input, channels, method=method, axis=axis)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize(('input1,input2,ratio,percentage_silence,'
                          'expected_output'), [
    (A, A, 0, 0, 2 * A),
    (A, A, _half_ratio, 0, 1.5 * A),
    (a11, a11, 0, 0, 2 * a11),
    (a11, a11, _half_ratio, 0, 1.5 * a11),
    (A, np.zeros_like(A), 0, 0, A),
    (np.zeros_like(A), A, 0, 0, A),
    (A, np.zeros_like(A), 0, 1, A),
])
def test_additivemix(input1, input2, ratio, percentage_silence,
                     expected_output):
    output = F.additive_mix(input1, input2, ratio)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize('input,axis,expected_output', [
    (A, None, A / np.max(A)),
    (A, -1, np.array([[a11 / max(a11), a12 / max(a12)],
                      [a21 / max(a21), a22 / max(a22)]])),
    (a11, None, a11 / np.max(a11)),
    (a11, -1, a11 / np.max(a11)),
    ([a11, a12], 1, np.array([a11 / max(a11), a12 / max(a12)])),
    ([[1, 4], [3, 2]], 0, np.array([[1 / 3, 4 / 4], [3 / 3, 2 / 4]])),
])
def test_normalize(input, axis, expected_output):
    output = F.normalize(input, axis=axis)
    assert np.array_equal(output, expected_output)


@pytest.mark.parametrize('input,axis,mean,std', [
    (A, None, True, True),
    (A, -1, True, True),
    (a11, None, True, True),
    (a11, -1, True, True),
    (A, -1, False, True),
    (A, -1, True, False),
    (A, -1, False, False),
])
def test_standardize(input, axis, mean, std):
    output = F.standardize(input, axis=axis, mean=mean, std=std)
    if mean:
        np.testing.assert_almost_equal(output.mean(axis=axis).mean(), 0)
    if std:
        np.testing.assert_almost_equal(output.std(axis=axis).mean(), 1)


@pytest.mark.parametrize('input,window_size,hop_size,axis', [
    (A, 4, 1, 2),
    pytest.param(A, 2048, 1024, 2, marks=xfail(raises=ValueError)),
    pytest.param(A, 3, 1, 2, marks=xfail),
    (np.random.normal(size=[2, 3, 16000]), 2048, 1024, 2),
    (np.random.normal(size=[2, 16000, 3]), 2048, 1024, 1),
    (np.random.normal(size=[16000, 2, 3]), 2048, 1024, 0),
    (np.random.normal(size=16000), 2048, 1024, 0),
])
def test_stft(input, window_size, hop_size, axis):
    expected_output = input
    samples = input.shape[axis]
    spectrogram = F.stft(input, window_size, hop_size, axis=axis)
    magnitude, phase = librosa.magphase(spectrogram)
    output = F.istft(spectrogram, window_size, hop_size, axis=axis)
    output = F.crop(output, (0, samples), axis=axis)
    np.testing.assert_almost_equal(output, expected_output, decimal=6)
