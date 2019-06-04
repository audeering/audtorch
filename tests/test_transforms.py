import pytest
import numpy as np
import scipy
import resampy
import librosa

import audtorch.transforms as transforms
import audtorch.transforms.functional as F


xfail = pytest.mark.xfail

a11 = np.array([1, 2, 3, 4], dtype=float)
a12 = np.array([5, 6, 7, 8], dtype=float)
a21 = np.array([9, 10, 11, 12], dtype=float)
a22 = np.array([13, 14, 15, 16], dtype=float)
ones = np.ones(4)
zeros = np.zeros(4)
A = np.array([[a11, a12], [a21, a22]])  # Tensor of shape (2, 2, 4)

# Ratio in dB to add two signals to yield 1.5 magnitude
_half_ratio = -10 * np.log10(0.5 ** 2)


def _mean(signal, axis):
    """Return mean along axis and preserve number of dimensions."""
    return np.expand_dims(np.mean(signal, axis=axis), axis=axis)


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
    t = transforms.Crop(idx, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,padding,value,axis,expected_output', [
    (A, 1, 0, -1, np.array([[_pad(a11, 1, 0), _pad(a12, 1, 0)],
                            [_pad(a21, 1, 0), _pad(a22, 1, 0)]])),
    (A, (0, 1), 1, 1, np.array([[a11, a12, ones], [a21, a22, ones]])),
    (A, (1, 0), 0, 0, np.array([[zeros, zeros], [a11, a12], [a21, a22]])),
    (a11, 1, 1, -1, _pad(a11, 1, 1)),
])
def test_pad(input, padding, value, axis, expected_output):
    t = transforms.Pad(padding, value=value, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,repetitions,axis', [
    (A, 2, -1),
    (A, 3, 0),
    (A, 4, 1),
    (a11, 1, 0)])
def test_replicate(input, repetitions, axis):
    expected_output = np.concatenate(tuple([input] * repetitions), axis)
    t = transforms.Replicate(repetitions, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,size,axis,method', [
    (A, 12, 0, 'pad'),
    (A, 12, 1, 'pad'),
    (A, 12, 2, 'pad'),
    (A, 12, 0, 'replicate'),
    (A, 12, 1, 'replicate'),
    (A, 12, 2, 'replicate')])
def test_expand(input, size, axis, method):
    t = transforms.Expand(size=size, axis=axis, method=method)
    if method == 'pad':
        assert np.array_equal(t(input), F.pad(
            input, (0, size - input.shape[axis]), axis=axis))
    else:
        assert np.array_equal(t(input), F.crop(F.replicate(
            input,
            repetitions=size // input.shape[axis] + 1,
            axis=axis), (0, size), axis=axis))


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
    t = transforms.Downmix(channels, method=method, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,channels,method,axis,expected_output', [
    (A, 2, 'mean', -2, A),
    (A, 3, 'mean', -2, np.hstack((A, _mean(A, axis=-2)))),
    (A, 3, 'zero', -2, np.hstack((A, [[zeros], [zeros]]))),
    (A, 3, 'repeat', -2, np.hstack((A, A[:, -1, None, :]))),
    (A, 3, 'mean', 0, np.vstack((A, _mean(A, axis=0)))),
    (a11, 2, 'repeat', -2, np.array([a11, a11])),
])
def test_upmix(input, channels, method, axis, expected_output):
    t = transforms.Upmix(channels, method=method, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,channels,axis,expected_output', [
    (A, 2, -2, A),
    (A, 3, -2, np.hstack((A, _mean(A, axis=-2)))),
    (A, 3, 0, np.vstack((A, _mean(A, axis=0)))),
    (A, 1, -2, _mean(A, axis=-2)),
    (A, 0, -2, np.empty((2, 0, 4))),  # empty array with correct shape
    (A, 1, 0, _mean(A, axis=0)),
])
def test_remix(input, channels, axis, expected_output):
    t = transforms.Remix(channels, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,axis,expected_output', [
    (A, None, A / np.max(A)),
    (A, -1, np.array([[a11 / max(a11), a12 / max(a12)],
                      [a21 / max(a21), a22 / max(a22)]])),
    (a11, None, a11 / np.max(a11)),
    (a11, -1, a11 / np.max(a11)),
])
def test_normalize(input, axis, expected_output):
    t = transforms.Normalize(axis=axis)
    assert np.array_equal(t(input), expected_output)


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
    t = transforms.Standardize(axis=axis)
    output = t(input)
    if mean:
        np.testing.assert_almost_equal(output.mean(axis=axis).mean(), 0)
    if std:
        np.testing.assert_almost_equal(output.std(axis=axis).mean(), 1)


@pytest.mark.parametrize('input,idx,axis', [
    (A, (0, 2), -1),
    (a11, (1, 2), -1),
])
def test_compose(input, idx, axis):
    t = transforms.Compose([transforms.Crop(idx, axis=axis),
                            transforms.Normalize(axis=axis)])
    expected_output = F.crop(input, idx, axis=axis)
    expected_output = F.normalize(expected_output, axis=axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,size,axis', [
    (A, 2, -1),
    (A, 1, -2),
    (A, 1, 0),
    (A, 0, -2),
    (a11, 3, -1),
])
def test_randomcrop(input, size, axis):
    t = transforms.RandomCrop(size, axis=axis)
    t.fix_randomization = True
    assert np.array_equal(t(input), t(input))
    assert np.array_equal(t(input), F.crop(input, t.idx, axis=t.axis))


@pytest.mark.parametrize('input,padding,value,axis', [
    (A, 1, 0, -1),
    (A, 2, 1, 1),
    (A, 0, 0, 0),
    (a11, 1, 1, -1),
])
def test_randompad(input, padding, value, axis):
    t = transforms.RandomPad(padding, value=value, axis=axis)
    t.fix_randomization = True
    assert np.array_equal(t(input), t(input))
    expected_output = F.pad(input, t.pad, value=t.value, axis=t.axis)
    assert np.array_equal(t(input), expected_output)


@pytest.mark.parametrize('input,input_sample_rate,output_sample_rate,axis', [
    (A, 4, 2, -1),
    (np.ones([4, 4, 2]), 4, 2, -2),
    (a11, 3, 2, -1),
])
@pytest.mark.parametrize('method', ['kaiser_best', 'kaiser_fast', 'scipy'])
def test_resample(input, input_sample_rate, output_sample_rate, method, axis):
    t = transforms.Resample(input_sample_rate, output_sample_rate,
                            method=method, axis=axis)
    output_length = int(input.shape[axis] * output_sample_rate
                        / float(input_sample_rate))
    print(input.shape)
    if method == 'scipy':
        expected_output = scipy.signal.resample(input, output_length,
                                                axis=axis)
    else:
        expected_output = resampy.resample(input, input_sample_rate,
                                           output_sample_rate, method=method,
                                           axis=axis)
    transformed_input = t(input)
    assert transformed_input.shape[axis] == output_length
    assert np.array_equal(transformed_input, expected_output)


@pytest.mark.parametrize('input,window_size,hop_size,axis', [
    (A, 4, 1, 2),
    pytest.param(A, 2048, 1024, 2, marks=xfail(raises=ValueError)),
    (np.random.normal(size=[2, 3, 16000]), 2048, 1024, 2),
    (np.random.normal(size=[2, 16000, 3]), 2048, 1024, 1),
    (np.random.normal(size=[16000, 2, 3]), 2048, 1024, 0),
    (np.random.normal(size=[16000, 2, 3]), 2048, 1024, 0),
    (np.random.normal(size=16000), 2048, 1024, 0),
])
def test_stft(input, window_size, hop_size, axis):
    t = transforms.Spectrogram(window_size, hop_size, axis=axis)
    spectrogram = F.stft(input, window_size, hop_size, axis=axis)
    magnitude, phase = librosa.magphase(spectrogram)
    assert np.array_equal(t(input), magnitude)
    assert np.array_equal(t.phase, phase)


@pytest.mark.parametrize('input,magnitude_boost', [
    (A, 1e-07),
    pytest.param(A, -1e-07, marks=xfail(raises=ValueError)),
    pytest.param(A, -1, marks=xfail(raises=ValueError)),
    (np.random.normal(size=[2, 3, 16000]), 1e-03),
    (np.random.normal(size=[2, 16000, 3]), 1e-07),
    (np.random.normal(size=[16000, 2, 3]), 1e-07),
    (np.random.normal(size=16000), 1e-07),
])
def test_log(input, magnitude_boost):
    input = input + abs(input.min())
    t_log = transforms.Log(magnitude_boost=magnitude_boost)
    assert np.array_equal(t_log(input), np.log(input + magnitude_boost))


@pytest.mark.parametrize('signal1,signal2,ratio,'
                         'percentage_silence,expected_signal', [
                             (A, A, 0, 0, 2 * A),
                             (A, A, _half_ratio, 0, 1.5 * A),
                             (a11, a11, 0, 0, 2 * a11),
                             (a11, a11, _half_ratio, 0, 1.5 * a11),
                             (A, np.zeros_like(A), 0, 0, A),
                             (np.zeros_like(A), A, 0, 0, A),
                             (A, np.zeros_like(A), 0, 1, A),
                         ])
def test_randomadditivemix(signal1, signal2, ratio, percentage_silence,
                           expected_signal):
    mixed_signal = F.additive_mix(signal1, signal2, ratio)
    assert np.array_equal(mixed_signal, expected_signal)
    augmentation_data_set = [[signal2]]
    t = transforms.RandomAdditiveMix(dataset=augmentation_data_set,
                                     ratios=[ratio],
                                     percentage_silence=percentage_silence)
    assert np.array_equal(t(signal1), expected_signal)
    t = transforms.RandomAdditiveMix(dataset=augmentation_data_set)
    transform = t(signal1)
    functional = F.additive_mix(signal1, signal2, t.ratio)
    assert np.array_equal(transform, functional)
