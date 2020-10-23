import librosa
import numpy as np
import torch

from .. import utils


def crop(
        signal,
        idx,
        *,
        axis=-1,
):
    r"""Crop signal along an axis.

    Args:
        signal (numpy.ndarray): audio signal
        idx (int or tuple): first (and last) index to return
        axis (int, optional): axis along to crop. Default: `-1`

    Note:
        Indexing from the end with `-1`, `-2`, ... is allowed. But you cannot
        use `-1` in the second part of the tuple to specify the last entry.
        Instead you have to write `(-2, signal.shape[axis])` to get the last
        two entries of `axis`, or simply `-1` if you only want to get the last
        entry.

    Returns:
        numpy.ndarray: cropped signal

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> crop(a, 1)
        array([[2],
               [4]])

    """
    # Ensure idx is iterate able
    if isinstance(idx, int):
        idx = [idx]
    # Allow for -1 like syntax for index
    length = signal.shape[axis]
    idx = [length + i if i < 0 else i for i in idx]
    # Add stop index for single values
    if len(idx) == 1:
        idx = [idx[0], idx[0] + 1]

    # Split into three parts and return middle one
    return np.split(signal, idx, axis=axis)[1]


def pad(
        signal,
        padding,
        *,
        value=0,
        axis=-1,
):
    r"""Pad signal along an axis.

    If padding is an integer it pads equally on the left and right of the
    signal. If padding is a tuple with two entries it uses the first for the
    left side and the second for the right side.

    Args:
        signal (numpy.ndarray): audio signal
        padding (int or tuple): padding to apply on the left and right
        value (float, optional): value to pad with. Default: `0`
        axis (int, optional): axis along which to pad. Default: `-1`

    Returns:
        numpy.ndarray: padded signal

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> pad(a, (0, 1))
        array([[1, 2, 0],
               [3, 4, 0]])

    """
    padding = utils.to_tuple(padding)
    dimensions = np.ndim(signal)
    pad = [(0, 0) for _ in range(dimensions)]  # no padding for all axes
    pad[axis] = padding                        # padding along selected axis

    return np.pad(signal, pad, 'constant', constant_values=value)


def replicate(
        signal,
        repetitions,
        *,
        axis=-1,
):
    r"""Replicate signal along an axis.

    Args:
        signal (numpy.ndarray): audio signal
        repetitions (int): number of times to replicate signal
        axis (int, optional): axis along which to replicate. Default: `-1`

    Returns:
        numpy.ndarray: replicated signal

    Example:
        >>> a = np.array([1, 2, 3])
        >>> replicate(a, 3)
        array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    """
    reps = [1 for _ in range(len(signal.shape))]
    reps[axis] = repetitions
    return np.tile(signal, reps)


def mask(
        signal,
        num_blocks,
        max_width,
        *,
        value=0.,
        axis=-1,
):
    r"""Randomly mask signal along axis.

    Args:
        signal (torch.Tensor): audio signal
        num_blocks (int): number of mask blocks
        max_width (int): maximum size of block
        value (float, optional): mask value. Default: `0.`
        axis (int, optional): axis along which to mask. Default: `-1`

    Returns:
        torch.Tensor: masked signal

    """
    signal_size = signal.shape[axis]
    # add 1 to `max_width` to include value `max_width` in sampling
    widths = torch.randint(low=1, high=max_width + 1, size=(num_blocks,))
    start = torch.LongTensor(
        [torch.randint(0, signal_size - widths[i].item(), (1,))
         for i in range(num_blocks)])

    for i, s in enumerate(start):
        signal.narrow(start=s.item(),
                      length=widths[i].item(),
                      dim=axis).fill_(value)
    return signal


def downmix(
        signal,
        channels,
        *,
        method='mean',
        axis=-2,
):
    r"""Downmix signal to the provided number of channels.

    The downmix is done by one of these methods:

        * ``'mean'`` replace last desired channel by mean across itself and
          all remaining channels
        * ``'crop'`` drop all remaining channels

    Args:
        signal (numpy.ndarray): audio signal
        channels (int): number of desired channels
        method (str, optional): downmix method. Default: `'mean'`
        axis (int, optional): axis to downmix. Default: `-2`

    Returns:
        numpy.ndarray: reshaped signal

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> downmix(a, 1)
        array([[2, 3]])

    """
    input_channels = np.atleast_2d(signal).shape[axis]

    if input_channels <= channels:
        return signal

    if method == 'mean':
        downmix = crop(signal, (channels - 1, input_channels), axis=axis)
        downmix = np.mean(downmix, axis=axis)
        signal = np.insert(signal, channels - 1, downmix, axis=axis)
    elif method == 'crop':
        pass
    else:
        raise TypeError(f'Method {method} not supported.')

    signal = crop(signal, (0, channels), axis=axis)
    return signal


def upmix(
        signal,
        channels,
        *,
        method='mean',
        axis=-2,
):
    r"""Upmix signal to the provided number of channels.

    The upmix is achieved by adding the same signal in the additional channels.
    The fixed signal is calculated by one of the following methods:

        * ``'mean'`` mean across all input channels
        * ``'zero'`` zeros
        * ``'repeat'`` last input channel

    Args:
        signal (numpy.ndarray): audio signal
        channels (int): number of desired channels
        method (str, optional): upmix method. Default: `'mean'`
        axis (int, optional): axis to upmix. Default: `-2`

    Returns:
        numpy.ndarray: reshaped signal

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> upmix(a, 3)
        array([[1., 2.],
               [3., 4.],
               [2., 3.]])

    """
    input_channels = np.atleast_2d(signal).shape[axis]

    if input_channels >= channels:
        return signal

    signal = np.atleast_2d(signal)
    if method == 'mean':
        upmix = np.mean(signal, axis=axis)
        upmix = np.expand_dims(upmix, axis=axis)
    elif method == 'zero':
        upmix = np.zeros(signal.shape)
        upmix = crop(upmix, -1, axis=axis)
    elif method == 'repeat':
        upmix = crop(signal, -1, axis=axis)
    else:
        raise TypeError(f'Method {method} not supported.')

    upmix = np.repeat(upmix, channels - input_channels, axis=axis)
    return np.concatenate((signal, upmix), axis=axis)


def additive_mix(
        signal1,
        signal2,
        ratio,
):
    r"""Mix two signals additively by given ratio.

    If the power of one of the signals is below 1e-7, the signals are added
    without adjusting the signal-to-noise ratio.

    Args:
        signal1 (numpy.ndarray): audio signal
        signal2 (numpy.ndarray): audio signal
        ratio (int): ratio in dB of the second signal compared to the first one

    Returns:
        numpy.ndarray: mixture

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> additive_mix(a, a, -10 * np.log10(0.5 ** 2))
        array([[1.5, 3. ],
               [4.5, 6. ]])

    """
    if signal1.shape != signal2.shape:
        raise ValueError(
            f'Shape of signal1 ({signal1.shape}) '
            f'and signal2 ({signal2.shape}) do not match'
        )
    # If one of the signals includes only silence, don't apply SNR
    tol = 1e-7
    if utils.power(signal1) < tol or utils.power(signal2) < tol:
        scaling_factor = 1
    else:
        scaling_factor = (utils.power(signal1)
                          / utils.power(signal2)
                          * 10 ** (-ratio / 10))
    return signal1 + np.sqrt(scaling_factor) * signal2


def normalize(
        signal,
        *,
        axis=None,
):
    r"""Normalize signal.

    Ensure the maximum of the absolute value of the signal is 1.

    Note:
        The signal will never be divided by a number smaller than 1e-7.
        Meaning signals which are nearly silent are only slightly
        amplified.

    Args:
        signal (numpy.ndarray): audio signal
        axis (int, optional): normalize only along the given axis.
            Default: `None`

    Returns:
        numpy.ndarray: normalized signal

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> normalize(a)
        array([[0.25, 0.5 ],
               [0.75, 1.  ]])

    """
    if axis is not None:
        peak = np.expand_dims(np.amax(np.abs(signal), axis=axis), axis=axis)
    else:
        peak = np.amax(np.abs(signal))
    return signal / np.maximum(peak, 1e-7)


def standardize(
        signal,
        *,
        mean=True,
        std=True,
        axis=None,
):
    r"""Standardize signal.

    Ensure the signal has a mean value of 0 and a variance of 1.

    Note:
        The signal will never be divided by a variance smaller than 1e-7.

    Args:
        signal (numpy.ndarray): audio signal
        mean (bool, optional): apply mean centering. Default: `True`
        std (bool, optional): normalize by standard deviation. Default: `True`
        axis (int, optional): standardize only along the given axis.
            Default: `None`

    Returns:
        numpy.ndarray: standardized signal

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> standardize(a)
        array([[-1.34164079, -0.4472136 ],
               [ 0.4472136 ,  1.34164079]])

    """
    if mean:
        signal_mean = np.mean(signal, axis=axis)
        if axis is not None:
            signal_mean = np.expand_dims(signal_mean, axis=axis)
        signal = signal - signal_mean
    if std:
        signal_std = np.std(signal, axis=axis)
        if axis is not None:
            signal_std = np.expand_dims(signal_std, axis=axis)
        signal = signal / np.maximum(signal_std, 1e-7)
    return signal


def stft(
        signal,
        window_size,
        hop_size,
        *,
        fft_size=None,
        window='hann',
        axis=-1,
):
    r"""Short-time Fourier transform.

    The Short-time Fourier transform (STFT) is calculated by using librosa.
    It returns an array with the same shape as the input array, besides the
    axis chosen for STFT calculation is replaced by the two new ones of the
    spectrogram.

    The chosen FFT size is set identical to `window_size`.

    Args:
        signal (numpy.ndarray): audio signal
        window_size (int): size of STFT window in samples
        hop_size (int): size of STFT window hop in samples
        window (str, tuple, number, function, or numpy.ndarray, optional): type
            of STFT window. Default: `hann`
        axis (int, optional): axis of STFT calculation. Default: `-1`

    Returns:
        numpy.ndarray: complex spectrogram with the shape of its last two
        dimensions as `(window_size/2 + 1,
        np.ceil((len(signal) + window_size/2) / hop_size))`

    Example:
        >>> a = np.array([1., 2., 3., 4.])
        >>> stft(a, 2, 1)
        array([[ 1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  3.+0.j],
               [-1.+0.j, -2.+0.j, -3.+0.j, -4.+0.j, -3.+0.j]])

    """
    samples = signal.shape[axis]
    if samples < window_size:
        raise ValueError(
            f'`signal` of length {samples} needs to be at least '
            f'as long as the `window_size` of {window_size}'
        )

    # Pad to ensure same signal length after reconstruction
    # See discussion at https://github.com/librosa/librosa/issues/328
    signal = pad(signal, (0, np.mod(samples, hop_size)), value=0, axis=axis)
    if fft_size is None:
        fft_size = window_size
    fft_config = dict(n_fft=fft_size, hop_length=hop_size,
                      win_length=window_size, window=window)
    spectrogram = np.apply_along_axis(librosa.stft, axis, signal, **fft_config)
    return spectrogram


def istft(
        spectrogram,
        window_size,
        hop_size,
        *,
        window='hann',
        axis=-2,
):
    r"""Inverse Short-time Fourier transform.

    The inverse Short-time Fourier transform (iSTFT) is calculated by using
    librosa.
    It handles multi-dimensional inputs, but assumes that the two spectrogram
    axis are beside each other, starting with the axis corresponding to
    frequency bins.
    The returned audio signal has one dimension less than the spectrogram.

    Args:
        spectrogram (numpy.ndarray): complex spectrogram
        window_size (int): size of STFT window in samples
        hop_size (int): size of STFT window hop in samples
        window (str, tuple, number, function, or numpy.ndarray, optional): type
            of STFT window. Default: `hann`
        axis (int, optional): axis of frequency bins of the spectrogram. Time
            bins are expected at `axis + 1`. Default: `-2`

    Returns:
        numpy.ndarray: signal with shape `(number_of_time_bins * hop_size -
        window_size/2)`

    Example:
        >>> a = np.array([1., 2., 3., 4.])
        >>> D = stft(a, 4, 1)
        >>> istft(D, 4, 1)
        array([1., 2., 3., 4.])

    """
    if axis == -1:
        raise ValueError('`axis` of spectrogram frequency bins cannot be -1')

    ifft_config = dict(hop_length=hop_size, win_length=window_size,
                       window=window)
    # Size of frequency and time axis
    f = spectrogram.shape[axis]
    t = spectrogram.shape[axis + 1]
    # Reshape the two axes of spectrogram into one axis
    shape_before = spectrogram.shape[:axis]
    shape_after = spectrogram.shape[axis:][2:]
    D = np.reshape(spectrogram, [*shape_before, f * t, *shape_after])
    # Adjust negative axis values as the second spectrogram axis was removed
    if axis < -1:
        axis += 1
    # iSTFT along the axis
    signal = np.apply_along_axis(_istft, axis, D, f, t, **ifft_config)
    # Remove padding that was added for STFT
    samples = signal.shape[axis]
    signal = crop(signal, (0, samples - np.mod(samples, hop_size)), axis=axis)
    return signal


def _istft(spectrogram, frequency_bins, time_bins, **config):
    """Inverse Short-time Fourier transform from a single axis.

    Time and frequency bins have to be provided in a single vector. This allows
    effective computation using `numpy.apply_along_axis`.

    Args:
        spectrogram (numpy.array): one dimensional vector
        frequency_bins (int): number of frequency bins
        time_bins (int): number of time bins
        **config (dict, optional): keyword arguments for librosa.istft

    Returns:
        numpy.array: time-series

    """
    # Reshape to [frequency_bins, time_bins] as expected by librosa
    spectrogram = np.reshape(spectrogram, [frequency_bins, time_bins])
    return librosa.istft(spectrogram, **config)
