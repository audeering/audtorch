import random
from warnings import warn

import numpy as np
import resampy
try:
    import librosa
except ImportError:
    librosa = None
try:
    import scipy
except ImportError:
    scipy = None

from . import functional as F


class Compose(object):
    """Compose several transforms together.

    Args:
        transforms (list of object): list of transforms to compose

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = Compose([Crop(-1), Pad(1)])
        >>> print(t)
        Compose(
            Crop(idx=-1, axis=-1)
            Pad(padding=1, value=0, axis=-1)
        )
        >>> t(a)
        array([[0, 2, 0],
               [0, 4, 0]])

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal):
        for t in self.transforms:
            signal = t(signal)
        return signal

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Crop(object):
    r"""Crop along an axis.

    * :attr:`idx` controls the index for cropping
    * :attr:`axis` controls axis of cropping

    Args:
        idx (int or tuple): first (and last) index to return
        axis (int, optional): axis along to crop. Default: `-1`

    Note:
        Indexing from the end with `-1`, `-2`, ... is allowed. But you cannot
        use `-1` in the second part of the tuple to specify the last entry.
        Instead you have to write `(-2, signal.shape[axis])` to get the last
        two entries of `axis`, or simply `-1` if you only want to get the last
        entry.

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to crop and :math:`N_\text{out}` is the
          output length, which is :math:`1` for an integer as `idx` and
          :math:`\text{idx[1]} - \text{idx[0]}` for a tuple with positive
          entries as `idx`.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = Crop(1, axis=1)
        >>> print(t)
        Crop(idx=1, axis=1)
        >>> t(a)
        array([[2],
               [4]])

    """

    def __init__(self, idx, *, axis=-1):
        super().__init__()
        self.idx = idx
        self.axis = axis

    def __call__(self, signal):
        return F.crop(signal, self.idx, axis=self.axis)

    def __repr__(self):
        options = 'idx={0}, axis={1}'.format(self.idx, self.axis)
        return '{0}({1})'.format(self.__class__.__name__, options)


class RandomCrop(object):
    r"""Random crop of specified width along an axis.

    If the signal is too short it is padded by trailing zeros first or
    replicated to fit specified size.

    If the signal is shorter than the desired length, it can be expanded by
    one of these methods:

        * ``'pad'`` expand the signal by adding trailing zeros
        * ``'replicate'`` first replicate the signal so that it matches or
          exceeds the specified size

    * :attr:`size` controls the size of output signal
    * :attr:`method` holds expansion method
    * :attr:`axis` controls axis of cropping
    * :attr:`fix_randomization` controls the randomness

    Args:
        size (int): desired width of spectrogram in samples
        method (str, optional): expansion method. Default: `pad`
        axis (int, optional): axis along to crop. Default: `-1`
        fix_randomization (bool, optional): fix random selection between
            different calls of transform. Default: `False`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to crop and :math:`N_\text{out}` is the
          output length as given by `size`.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> random.seed(0)
        >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> t = RandomCrop(2)
        >>> print(t)
        RandomCrop(size=2, method=pad, axis=-1)
        >>> t(a)
        array([[2, 3],
               [6, 7]])

    """

    def __init__(self, size, *, method='pad', axis=-1,
                 fix_randomization=False):
        super().__init__()
        self.size = size
        self.axis = axis
        self.fix_randomization = fix_randomization
        self.expand = Expand(size=0, axis=self.axis, method=method)
        self.idx = [0, 0]
        self.method = method

    def __call__(self, signal):
        input_size = signal.shape[self.axis]
        output_size = self.size

        # Pad or replicate if signal is too short
        if input_size < output_size:
            self.expand.size = output_size
            signal = self.expand(signal)
            input_size = output_size

        # Pad if random crop parameter is fixed and signal is too short
        if self.fix_randomization and input_size < self.idx[1]:
            self.expand.size = self.idx[1]
            signal = self.expand(signal)
            input_size = self.idx[1]

        if not self.fix_randomization:
            self.idx = self.random_index(input_size, output_size)

        return F.crop(signal, self.idx, axis=self.axis)

    @staticmethod
    def random_index(input_size, output_size):
        """Random index for crop.

        Args:
            input_size (int): input signal size
            output_size (int): expected output size

        Returns:
            tuple: random index for cropping

        """
        start = random.randint(0, input_size - output_size)
        return (start, start + output_size)

    def __repr__(self):
        options = 'size={0}, method={1}, axis={2}'.format(
            self.size, self.method, self.axis)
        if self.fix_randomization:
            options += ', fix_randomization=True'
        return '{0}({1})'.format(self.__class__.__name__, options)


class Pad(object):
    r"""Pad along an axis.

    If padding is an integer it pads equally on the left and right of the
    signal. If padding is a tuple with two entries it uses the first for the
    left side and the second for the right side.

    * :attr:`padding` controls the padding to be applied
    * :attr:`value` controls the value used for padding
    * :attr:`axis` controls the axis of padding

    Args:
        padding (int or tuple): padding to apply on the left and right
        value (float, optional): value to pad with. Default: `0`
        axis (int, optional): axis along to pad. Default: `-1`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to pad and :math:`N_\text{out} =
          N_\text{in} + \sum \text{padding}` is the output length.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = Pad((0, 1))
        >>> print(t)
        Pad(padding=(0, 1), value=0, axis=-1)
        >>> t(a)
        array([[1, 2, 0],
               [3, 4, 0]])

    """

    def __init__(self, padding, *, value=0, axis=-1):
        super().__init__()
        self.padding = padding
        self.value = value
        self.axis = axis

    def __call__(self, signal):
        return F.pad(signal, self.padding, value=self.value, axis=self.axis)

    def __repr__(self):
        options = ('padding={0}, value={1}, axis={2}'
                   .format(self.padding, self.value, self.axis))
        return '{0}({1})'.format(self.__class__.__name__, options)


class RandomPad(object):
    r"""Random pad along an axis.

    It splits the padding value randomly between the left and right of the
    signal along the specified axis.

    * :attr:`padding` controls the size of padding to be applied
    * :attr:`value` controls the value used for padding
    * :attr:`axis` controls the axis of padding
    * :attr:`fix_randomization` controls the randomness

    Args:
        padding (int): padding to apply randomly split on the left and right
        value (float, optional): value to pad with. Default: `0`
        axis (int, optional): axis along to pad. Default: `-1`
        fix_randomization (bool, optional): fix random selection between
            different calls of transform. Default: `False`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to pad and :math:`N_\text{out} =
          N_\text{in} + \sum \text{padding}` is the output length.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> random.seed(0)
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = RandomPad(1)
        >>> print(t)
        RandomPad(padding=1, value=0, axis=-1)
        >>> t(a)
        array([[0, 1, 2],
               [0, 3, 4]])

    """

    def __init__(self, padding, *, value=0, axis=-1, fix_randomization=False):
        super().__init__()
        self.padding = padding
        self.value = value
        self.axis = axis
        self.fix_randomization = fix_randomization
        self.pad = None

    def __call__(self, signal):
        if not self.pad or not self.fix_randomization:
            self.pad = self.random_split(self.padding)
        return F.pad(signal, self.pad, value=self.value, axis=self.axis)

    def __repr__(self):
        options = ('padding={0}, value={1}, axis={2}'
                   .format(self.padding, self.value, self.axis))
        if self.fix_randomization:
            options += ', fix_randomization=True'
        return '{0}({1})'.format(self.__class__.__name__, options)

    @staticmethod
    def random_split(number):
        """Split number randomly into two which sum up to number.

        Args:
            number (int): input number to be split

        Returns:
            tuple: randomly splitted number

        """
        left = random.randint(0, number)
        return (left, number - left)


class Replicate(object):
    r"""Replicate along an axis.

    * :attr:`repetitions` controls number of signal replications
    * :attr:`axis` controls the axis of replication

    Args:
        repetitions (int or tuple): number of times to replicate signal
        axis (int, optional): axis along which to replicate. Default: `-1`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to replicate and :math:`N_\text{out} =
          N_\text{in} \cdot \text{repetitions}` is the output length.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2, 3]])
        >>> t = Replicate(3)
        >>> print(t)
        Replicate(repetitions=3, axis=-1)
        >>> t(a)
        array([[1, 2, 3, 1, 2, 3, 1, 2, 3]])

    """

    def __init__(self, repetitions, *, axis=-1):
        super().__init__()
        self.repetitions = repetitions
        self.axis = axis

    def __call__(self, signal):
        return F.replicate(signal, self.repetitions, axis=self.axis)

    def __repr__(self):
        options = ('repetitions={0}, axis={1}'
                   .format(self.repetitions, self.axis))
        return '{0}({1})'.format(self.__class__.__name__, options)


class RandomReplicate(object):
    r"""Replicate by a random number of times along an axis.

    * :attr:`repetitions` holds number of times to replicate signal
    * :attr:`axis` controls the axis of replication
    * :attr:`fix_randomization` controls the randomness

    Args:
        max_repetitions (int, optional): controls the maximum number of times
            a signal is allowed to be replicated. Default: `100`
        axis (int, optional): axis along which to pad. Default: `-1`
        fix_randomization (bool, optional): fix random selection between
            different calls of transform. Default: `False`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to pad and :math:`N_\text{out} =
          N_\text{in} \cdot \text{repetitions}` is the output length.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> random.seed(0)
        >>> a = np.array([1, 2, 3])
        >>> t = RandomReplicate(max_repetitions=3)
        >>> print(t)
        RandomReplicate(max_repetitions=3, repetitions=None, axis=-1)
        >>> t(a)
        array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    """

    def __init__(self, *, max_repetitions=100, axis=-1,
                 fix_randomization=False):
        super().__init__()
        self.repetitions = None
        self.max_repetitions = max_repetitions
        self.axis = axis
        self.fix_randomization = fix_randomization

    def __call__(self, signal):
        if self.repetitions is None or not self.fix_randomization:
            self.repetitions = random.randint(0, self.max_repetitions)
        return F.replicate(signal, self.repetitions, axis=self.axis)

    def __repr__(self):
        options = ('max_repetitions={0}, repetitions={1}, axis={2}'
                   .format(self.max_repetitions, self.repetitions, self.axis))
        if self.fix_randomization:
            options += ', fix_randomization=True'
        return '{0}({1})'.format(self.__class__.__name__, options)


class Expand(object):
    r"""Expand signal.

    Ensures that the signal matches the desired output size by padding or
    replicating it.

    * :attr:`size` controls the size of output signal
    * :attr:`method` controls whether to replicate signal or pad it
    * :attr:`axis` controls axis of expansion

    The expansion is done by one of these methods:

        * ``'pad'`` expand the signal by adding trailing zeros
        * ``'replicate'`` replicate the signal to match the specified size.
          If result exceeds specified size after replication, the signal will
          then be cropped

    Args:
        size (int): desired length of output signal in samples
        method (str, optional): expansion method. Default: `pad`
        axis (int, optional): axis along to crop. Default: `-1`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_\text{out}, *)`, where :math:`N_\text{in}` is
          the input length of the axis to expand and :math:`N_\text{out}` is
          the output length as given by `size`.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2, 3]])
        >>> t = Expand(6)
        >>> print(t)
        Expand(size=6, method=pad, axis=-1)
        >>> t(a)
        array([[1, 2, 3, 0, 0, 0]])

    """

    def __init__(self, size, *, method='pad', axis=-1):
        super().__init__()
        self.size = size
        self.method = method
        self.axis = axis

    def __call__(self, signal):
        input_size = signal.shape[self.axis]
        output_size = self.size

        if input_size < output_size:
            if self.method == 'replicate':
                signal = F.replicate(signal, output_size // input_size + 1,
                                     axis=self.axis)
                signal = F.crop(signal, (0, output_size), axis=self.axis)
            elif self.method == 'pad':
                signal = F.pad(signal, (0, output_size - input_size),
                               axis=self.axis)
        return signal

    def __repr__(self):
        options = 'size={0}, method={1}, axis={2}'.format(
            self.size, self.method, self.axis)
        return '{0}({1})'.format(self.__class__.__name__, options)


class Downmix(object):
    r"""Downmix to the provided number of channels.

    The downmix is done by one of these methods:

        * ``'mean'`` replace last desired channel by mean across itself and
          all remaining channels
        * ``'crop'`` drop all remaining channels

    * :attr:`channels` controls the number of desired channels
    * :attr:`method` controls downmixing method
    * :attr:`axis` controls axis of downmix

    Args:
        channels (int): number of desired channels
        method (str, optional): downmix method. Default: `'mean'`
        axis (int, optional): axis to downmix. Default: `-2`

    Shape:
        - Input: :math:`(*, C_\text{in}, *)`
        - Output: :math:`(*, C_\text{out}, *)`, where :math:`C_\text{in}` is
          the number of input channels and :math:`C_\text{out}` is the number
          of output channels as given by `channels`. :math:`*` can be any
          additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = Downmix(1, axis=0)
        >>> print(t)
        Downmix(channels=1, method=mean, axis=0)
        >>> t(a)
        array([[2, 3]])

    """

    def __init__(self, channels, *, method='mean', axis=-2):
        super().__init__()
        self.channels = channels
        self.method = method
        self.axis = axis

    def __call__(self, signal):
        return F.downmix(signal, self.channels, method=self.method,
                         axis=self.axis)

    def __repr__(self):
        options = ('channels={0}, method={1}, axis={2}'
                   .format(self.channels, self.method, self.axis))
        return '{0}({1})'.format(self.__class__.__name__, options)


class Upmix(object):
    r"""Upmix to the provided number of channels.

    The upmix is achieved by adding the same signal in the additional channels.
    This signal is calculated by one of the following methods:

        * ``'mean'`` mean across all input channels
        * ``'zero'`` zeros
        * ``'repeat'`` last input channel

    * :attr:`channels` controls the number of desired channels
    * :attr:`method` controls downmixing method
    * :attr:`axis` controls axis of upmix

    Args:
        channels (int): number of desired channels
        method (str, optional): upmix method. Default: `'mean'`
        axis (int, optional): axis to upmix. Default: `-2`

    Shape:
        - Input: :math:`(*, C_\text{in}, *)`
        - Output: :math:`(*, C_\text{out}, *)`, where :math:`C_\text{in}` is
          the number of input channels and :math:`C_\text{out}` is the number
          of output channels as given by `channels`.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = Upmix(3, axis=0)
        >>> print(t)
        Upmix(channels=3, method=mean, axis=0)
        >>> t(a)
        array([[1., 2.],
               [3., 4.],
               [2., 3.]])

    """

    def __init__(self, channels, *, method='mean', axis=-2):
        super().__init__()
        self.channels = channels
        self.method = method
        self.axis = axis

    def __call__(self, signal):
        return F.upmix(signal, self.channels, method=self.method,
                       axis=self.axis)

    def __repr__(self):
        options = ('channels={0}, method={1}, axis={2}'
                   .format(self.channels, self.method, self.axis))
        return '{0}({1})'.format(self.__class__.__name__, options)


class Remix(object):
    r"""Remix to the provided number of channels.

    The remix is achieved by repeating the mean of all other channels or by
    replacing the last desired channel by the mean across all channels.

    It is internally achieved by running :class:`Upmix` or :class:`Downmix`
    with method `mean`.

    * :attr:`channels` controls the number of desired channels
    * :attr:`axis` controls axis of upmix

    Args:
        channels (int): number of desired channels
        axis (int, optional): axis to upmix. Default: `-2`

    Shape:
        - Input: :math:`(*, C_\text{in}, *)`
        - Output: :math:`(*, C_\text{out}, *)`, where :math:`C_\text{in}` is
          the number of input channels and :math:`C_\text{out}` is the number
          of output channels as given by `channels`.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> t = Remix(3, axis=0)
        >>> print(t)
        Remix(channels=3, axis=0)
        >>> t(a)
        array([[1., 2.],
               [3., 4.],
               [2., 3.]])

    """

    def __init__(self, channels, *, method='mean', axis=-2):
        super().__init__()
        self.channels = channels
        self.axis = axis

    def __call__(self, signal):
        if self.channels > np.atleast_2d(signal).shape[self.axis]:
            signal = F.upmix(signal, self.channels, method='mean',
                             axis=self.axis)
        elif self.channels < np.atleast_2d(signal).shape[self.axis]:
            signal = F.downmix(signal, self.channels, method='mean',
                               axis=self.axis)
        return signal

    def __repr__(self):
        options = 'channels={}, axis={}'.format(self.channels, self.axis)
        return '{}({})'.format(self.__class__.__name__, options)


class Normalize(object):
    r"""Normalize signal.

    Ensure the maximum of the absolute value of the signal is 1.

    * :attr:`axis` controls axis for normalization

    Args:
        axis (int, optional): axis for normalization. Default: `-1`

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*)`, where :math:`*` can be any number of dimensions.

    Example:
        >>> a = np.array([1, 2, 3, 4])
        >>> t = Normalize()
        >>> print(t)
        Normalize(axis=-1)
        >>> t(a)
        array([0.25, 0.5 , 0.75, 1.  ])

    """

    def __init__(self, *, axis=-1):
        super().__init__()
        self.axis = axis

    def __call__(self, signal):
        return F.normalize(signal, axis=self.axis)

    def __repr__(self):
        options = 'axis={0}'.format(self.axis)
        return '{0}({1})'.format(self.__class__.__name__, options)


class Resample(object):
    """Resample to new sampling rate.

    The signal is resampled by one of the following methods.

        * ``'kaiser_best'`` as implemented by resampy
        * ``'kaiser_fast'`` as implemented by resampy
        * ``'scipy'`` uses scipy for resampling

    * :attr:`input_sampling_rate` controls input sample rate in Hz
    * :attr:`output_sampling_rate` controls output sample rate in Hz
    * :attr:`method` controls the resample method
    * :attr:`axis` controls axis for resampling

    Args:
        input_sampling_rate (int): input sample rate in Hz
        output_sampling_rate (int): output sample rate in Hz
        method (str, optional): resample method. Default: `kaiser_best`
        axis (int, optional): axis for resampling. Default: `-1`

    Note:
        If the default method `kaiser_best` is too slow for your purposes,
        you should try `scipy` instead. `scipy` is the fastest method, but
        might crash for very long signals.

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*)`, where :math:`*` can be any number of dimensions.

    Example:
        >>> a = np.array([1, 2, 3, 4])
        >>> t = Resample(4, 2)
        >>> print(t)
        Resample(input_sampling_rate=4, output_sampling_rate=2, method=kaiser_best, axis=-1)
        >>> t(a)
        array([0, 2])

    """  # noqa: E501

    def __init__(self, input_sampling_rate, output_sampling_rate, *,
                 method='kaiser_best', axis=-1):
        super().__init__()
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.method = method
        self.axis = axis

    def __call__(self, signal):
        if self.method == 'scipy':
            out_samples = int(signal.shape[self.axis]
                              * self.output_sampling_rate
                              / float(self.input_sampling_rate))
            signal = scipy.signal.resample(signal, out_samples, axis=self.axis)
        else:
            signal = resampy.resample(signal, self.input_sampling_rate,
                                      self.output_sampling_rate,
                                      method=self.method, axis=self.axis)
        return signal

    def __repr__(self):
        options = ('input_sampling_rate={0}, output_sampling_rate={1}, '
                   'method={2}, axis={3}'
                   .format(self.input_sampling_rate, self.output_sampling_rate,
                           self.method, self.axis))
        return '{0}({1})'.format(self.__class__.__name__, options)


class Spectrogram(object):
    r"""Spectrogram of an audio signal.

    The spectrogram is calculated by librosa and its magnitude is returned as
    real valued matrix.

    * :attr:`window_size` controls FFT window size in samples
    * :attr:`hop_size` controls STFT window hop size in samples
    * :attr:`window` controls window function of spectrogram computation
    * :attr:`axis` controls axis of spectrogram computation
    * :attr:`phase` holds the phase of the spectrogram

    Args:
        window_size (int): size of STFT window in samples
        hop_size (int): size of STFT window hop in samples
        window (str, tuple, number, function, or numpy.ndarray, optional): type
            of STFT window. Default: `hann`
        axis (int, optional): axis of STFT calculation. Default: `-1`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_f, N_t, *)`, where :math:`N_\text{in}` is
          the number of input samples and
          :math:`N_t = {\text{window\_size} \over 2} + 1` is the number of
          output samples along the time axis of the spectrogram, and
          :math:`N_f = \lceil {1 \over \text{hop\_size}} (N_\text{in} +
          {\text{window\_size} \over 2}) \rceil` is the number of output
          samples along the frequency axis of the spectrogram.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([1., 2., 3., 4.])
        >>> t = Spectrogram(2, 2)
        >>> print(t)
        Spectrogram(window_size=2, hop_size=2, axis=-1)
        >>> t(a)
        array([[1., 3., 3.],
               [1., 3., 3.]], dtype=float32)

    """

    def __init__(self, window_size, hop_size, *, window='hann', axis=-1):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = window
        self.axis = axis
        self.phase = []

    def __call__(self, signal):
        spectrogram = F.stft(signal, self.window_size, self.hop_size,
                             window=self.window, axis=self.axis)
        magnitude, self.phase = librosa.magphase(spectrogram)
        return magnitude

    def __repr__(self):
        options = ('window_size={0}, hop_size={1}, axis={2}'
                   .format(self.window_size, self.hop_size, self.axis))
        if self.window != 'hann':
            options += ', window={0}'.format(self.window)
        return '{0}({1})'.format(self.__class__.__name__, options)


class LogSpectrogram(object):
    r"""Logarithmic spectrogram of an audio signal.

    The spectrogram is calculated by librosa and its magnitude is returned as
    real valued matrix. If `normalize` is set to `True` the magnitude is
    z-normalized by subtracting its mean and dividing by its standard
    deviation.

    * :attr:`window_size` controls FFT window size in samples
    * :attr:`hop_size` controls STFT window hop size in samples
    * :attr:`window` controls window function of spectrogram computation
    * :attr:`normalize` controls normalization of spectrogram
    * :attr:`magnitude_boost` controls the positive value added to the
      magnitude of the spectrogram before applying the logarithmus
    * :attr:`axis` controls axis of spectrogram computation
    * :attr:`phase` holds the phase of the spectrogram

    Args:
        window_size (int): size of STFT window in samples
        hop_size (int): size of STFT window hop in samples
        window (str, tuple, number, function, or numpy.ndarray, optional): type
            of STFT window. Default: `hann`
        normalize (bool, optional): normalize spectrogram. Default: `False`
        magnitude_boost (float, optional): positive value added to the
            magnitude of the spectrogram before applying the logarithmus.
            Default: `1e-7`
        axis (int, optional): axis of STFT calculation. Default: `-1`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_f, N_t, *)`, where :math:`N_\text{in}` is
          the number of input samples and
          :math:`N_f = 1 + {\text{window\_size} \over 2}` is the number of
          output samples along the frequency axis of the spectrogram, and
          :math:`N_t = \lceil {1 \over \text{hop\_size}} (N_\text{in} +
          {\text{window\_size} \over 2}) \rceil` is the number of output
          samples along the time axis of the spectrogram.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([1., 2., 3., 4.])
        >>> t = LogSpectrogram(2, 2)
        >>> print(t)
        LogSpectrogram(window_size=2, hop_size=2, magnitude_boost=1e-07, axis=-1, normalize=False)
        >>> t(a)
        array([[1.1920928e-07, 1.0986123e+00, 1.0986123e+00],
               [1.1920928e-07, 1.0986123e+00, 1.0986123e+00]], dtype=float32)

    """  # noqa: E501

    def __init__(self, window_size, hop_size, *, window='hann',
                 normalize=False, magnitude_boost=1e-7, axis=-1):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.window = window
        self.normalize = normalize
        self.magnitude_boost = magnitude_boost
        self.axis = axis
        self.phase = []
        if self.normalize:
            warn('Spectrogram normalization is enabled. Do not combine it '
                 'with signal reconstruction as it will degrade the result.',
                 UserWarning)
        if self.magnitude_boost <= 0:
            raise ValueError('`magnitude_boost` has to be >0, but is {}'
                             .format(self.magnitude_boost))

    def __call__(self, signal):
        spectrogram = F.stft(signal, self.window_size, self.hop_size,
                             window=self.window, axis=self.axis)
        magnitude, self.phase = librosa.magphase(spectrogram)
        magnitude = np.log(magnitude + self.magnitude_boost)

        if self.normalize:
            magnitude -= magnitude.mean()
            magnitude /= magnitude.std()

        return magnitude

    def __repr__(self):
        options = ('window_size={}, hop_size={}, magnitude_boost={}, axis={}'
                   .format(self.window_size, self.hop_size,
                           self.magnitude_boost, self.axis))
        if not self.normalize:
            options += ', normalize=False'
        if self.window != 'hann':
            options += ', window={0}'.format(self.window)
        return '{0}({1})'.format(self.__class__.__name__, options)
