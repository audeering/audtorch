import random
from warnings import warn

import librosa
import numpy as np
import resampy
import torch
try:
    import scipy
except ImportError:
    scipy = None

from . import functional as F


class Compose(object):
    r"""Compose several transforms together.

    Args:
        transforms (list of object): list of transforms to compose
        fix_randomization (bool, optional): controls randomization of
            underlying transforms. Default: `False`

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

    def __init__(
            self,
            transforms,
            *,
            fix_randomization=False,
    ):
        self.transforms = transforms
        self.fix_randomization = fix_randomization

    def __call__(self, signal):
        for t in self.transforms:
            if hasattr(t, 'fix_randomization'):
                t.fix_randomization = self.fix_randomization
            signal = t(signal)
        return signal

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        if self.fix_randomization:
            format_string += ', fix_randomization=True'
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

    def __init__(
            self,
            idx,
            *,
            axis=-1,
    ):
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

    def __init__(
            self,
            size,
            *,
            method='pad',
            axis=-1,
            fix_randomization=False,
    ):
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

    def __init__(
            self,
            padding,
            *,
            value=0,
            axis=-1,
    ):
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

    def __init__(
            self,
            padding,
            *,
            value=0,
            axis=-1,
            fix_randomization=False,
    ):
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

    def __init__(
            self,
            repetitions,
            *,
            axis=-1,
    ):
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

    def __init__(
            self,
            *,
            max_repetitions=100,
            axis=-1,
            fix_randomization=False,
    ):
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

    def __init__(
            self,
            size,
            *,
            method='pad',
            axis=-1,
    ):
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


class RandomMask(object):
    r"""Randomly masks signal along axis.

    The signal is masked by multiple blocks (i.e. consecutive units) size of
    which is uniformly sampled given an upper limit on the block size.
    The algorithm for a single block is as follows:

        1. :math:`\text{width} ~ U[0, {\text{maximum\_width}}]`
        2. :math:`\text{start} ~ U[0, {\text{signal\_size}} - \text{width})`

    The number of blocks is approximated by the specified `coverage` of the
    masking and the average size of a block.

    * :attr:`coverage` controls how large the proportion of masking is
      relative to the signal size
    * :attr:`max_width` controls the maximum size of a masked block
    * :attr:`value` controls the value to mask the signal with
    * :attr:`axis` controls the axis to mask the signal along

    Args:
        coverage (float): proportion of signal to mask
        max_width (int): maximum block size. The unit depends on the signal
            and axis. See `MaskSpectrogramTime` and `MaskSpectrogramFrequency`
        value (float): mask value
        axis (int): axis to mask signal along

    Example:
        >>> a = torch.empty((1, 4, 10)).uniform_(1, 2)
        >>> t = RandomMask(0.1, max_width=1, value=0, axis=2)
        >>> print(t)
        RandomMask(coverage=0.1, max_width=1, value=0, axis=2)
        >>> len((t(a) == 0).nonzero())  # number of 0 elements
        4

    """
    def __init__(
            self,
            coverage,
            max_width,
            value,
            axis,
    ):
        self.coverage = coverage
        self.max_width = max_width
        self.value = value
        self.axis = axis

    def __call__(self, signal):

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)

        signal_size = signal.shape[self.axis]

        if signal_size < self.max_width:
            raise RuntimeError(
                'size along axis {} is smaller than `max_width`: '
                '{} < {}'.format(self.axis, signal_size, self.max_width))

        average_width = (self.max_width + 1) / 2
        num_blocks = int(signal_size * self.coverage / average_width)
        if self.coverage != 0.:
            num_blocks = max(1, num_blocks)

        return F.mask(signal, num_blocks, self.max_width, value=self.value,
                      axis=self.axis)

    def __repr__(self):
        options = 'coverage={0}, max_width={1}, value={2}, axis={3}'.format(
            self.coverage, self.max_width, self.value, self.axis)
        return '{0}({1})'.format(self.__class__.__name__, options)


class MaskSpectrogramTime(RandomMask):
    r"""Randomly masks spectrogram along time axis.

    See :class:`RandomMask` for more details.

    Note:
        The time axis is derived from `Spectrogram`'s output shape.

    Args:
        coverage (float): proportion of signal to mask
        max_width (int): maximum block size in number of samples. The default
            value corresponds to a time span of 0.1 seconds of a signal
            with `sr=16000` and stft-specifications of `window_size=320` and
            `hop_size=160`. Default: `11`
        value (float): mask value

    Example:
        >>> from librosa.display import specshow  # doctest: +SKIP
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        >>> a = torch.empty(65000).uniform_(-1, 1)
        >>> t = Compose([Spectrogram(320, 160), MaskSpectrogramTime(0.1)])
        >>> magnitude = t(a).squeeze().numpy()
        >>> specshow(np.log10(np.abs(magnitude) + 1e-4)) # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

    """
    def __init__(
            self,
            coverage,
            *,
            max_width=11,
            value=0,
    ):
        super().__init__(coverage, max_width, value, axis=-1)


class MaskSpectrogramFrequency(RandomMask):
    r"""Randomly masks spectrogram along frequency axis.

    See :class:`RandomMask` for more details.

    Note:
        The frequency axis is derived from `Spectrogram`'s output shape.

    Args:
        coverage (float): proportion of signal to mask
        max_width (int, optional): maximum block size in number of
            frequency bins. The default value corresponds to approximately
            5% of all frequency bins with stft-specifications of
            `window_size=320` and `hop_size=160`. Default: `8`
        value (float): mask value

    Example:
        >>> from librosa.display import specshow  # doctest: +SKIP
        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        >>> a = torch.empty(65000).uniform_(-1, 1)
        >>> t = Compose([Spectrogram(320, 160), MaskSpectrogramFrequency(0.1)])
        >>> magnitude = t(a).squeeze().numpy()
        >>> specshow(np.log10(np.abs(magnitude) + 1e-4)) # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

    """
    def __init__(
            self,
            coverage,
            *,
            max_width=8,
            value=0,
    ):
        super().__init__(coverage, max_width, value, axis=-2)


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

    def __init__(
            self,
            channels,
            *,
            method='mean',
            axis=-2,
    ):
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

    def __init__(
            self,
            channels,
            *,
            method='mean',
            axis=-2,
    ):
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

    def __init__(
            self,
            channels,
            *,
            method='mean',
            axis=-2,
    ):
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

    def __init__(
            self,
            *,
            axis=-1,
    ):
        super().__init__()
        self.axis = axis

    def __call__(self, signal):
        return F.normalize(signal, axis=self.axis)

    def __repr__(self):
        options = 'axis={0}'.format(self.axis)
        return '{0}({1})'.format(self.__class__.__name__, options)


class Standardize(object):
    r"""Standardize signal.

    Ensure the signal has a mean value of 0 and a variance of 1.

    * :attr:`mean` controls whether mean centering will be applied
    * :attr:`std` controls whether standard deviation normalization will be
      applied
    * :attr:`axis` controls axis for standardization

    Args:
        mean (bool, optional): apply mean centering. Default: `True`
        std (bool, optional): normalize by standard deviation. Default: `True`
        axis (int, optional): standardize only along the given axis.
            Default: `-1`

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*)`, where :math:`*` can be any number of dimensions.

    Example:
        >>> a = np.array([1, 2, 3, 4])
        >>> t = Standardize()
        >>> print(t)
        Standardize(axis=-1, mean=True, std=True)
        >>> t(a)
        array([-1.34164079, -0.4472136 ,  0.4472136 ,  1.34164079])

    """

    def __init__(
            self,
            *,
            mean=True,
            std=True,
            axis=-1,
    ):
        super().__init__()
        self.axis = axis
        self.mean = mean
        self.std = std

    def __call__(self, signal):
        return F.standardize(signal, axis=self.axis,
                             mean=self.mean, std=self.std)

    def __repr__(self):
        options = 'axis={0}'.format(self.axis)
        if self.mean:
            options += ', mean=True'
        if self.std:
            options += ', std=True'
        return '{0}({1})'.format(self.__class__.__name__, options)


class Resample(object):
    r"""Resample to new sampling rate.

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

    def __init__(
            self,
            input_sampling_rate,
            output_sampling_rate,
            *,
            method='kaiser_best',
            axis=-1,
    ):
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
    * :attr:`fft_size` controls number of frequency bins in STFT
    * :attr:`window` controls window function of spectrogram computation
    * :attr:`axis` controls axis of spectrogram computation
    * :attr:`phase` holds the phase of the spectrogram

    Args:
        window_size (int): size of STFT window in samples
        hop_size (int): size of STFT window hop in samples
        fft_size(int, optional): number of frequency bins in STFT. If `None`,
            then it defaults to `window_size`. Default: `None`
        window (str, tuple, number, function, or numpy.ndarray, optional): type
            of STFT window. Default: `hann`
        axis (int, optional): axis of STFT calculation. Default: `-1`

    Shape:
        - Input: :math:`(*, N_\text{in}, *)`
        - Output: :math:`(*, N_f, N_t, *)`, where :math:`N_\text{in}` is
          the number of input samples and
          :math:`N_f = {\text{window\_size} \over 2} + 1` is the number of
          output samples along the frequency axis of the spectrogram, and
          :math:`N_t = \lceil {1 \over \text{hop\_size}} (N_\text{in} +
          {\text{window\_size} \over 2}) \rceil` is the number of output
          samples along the time axis of the spectrogram.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> a = np.array([1., 2., 3., 4.])
        >>> t = Spectrogram(2, 2)
        >>> print(t)
        Spectrogram(window_size=2, hop_size=2, axis=-1)
        >>> t(a)
        array([[1., 3., 3.],
               [1., 3., 3.]])

    """

    def __init__(
            self,
            window_size,
            hop_size,
            *,
            fft_size=None,
            window='hann',
            axis=-1,
    ):
        super().__init__()
        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.window = window
        self.axis = axis
        self.phase = []

    def __call__(self, signal):
        spectrogram = F.stft(signal, self.window_size, self.hop_size,
                             fft_size=self.fft_size, window=self.window,
                             axis=self.axis)
        magnitude, self.phase = librosa.magphase(spectrogram)
        return magnitude

    def __repr__(self):
        options = ('window_size={0}, hop_size={1}, axis={2}'
                   .format(self.window_size, self.hop_size, self.axis))
        if self.window != 'hann':
            options += ', window={0}'.format(self.window)
        return '{0}({1})'.format(self.__class__.__name__, options)


class Log(object):
    r"""Logarithmic transform of an input signal.

    * :attr:`magnitude_boost` controls the non-negative value added to the
      magnitude of the signal before applying the logarithmus

    Args:
        magnitude_boost (float, optional): positive value added to the
            magnitude of the signal before applying the logarithmus. Default:
            `1e-7`

    Shape:
        - Input: :math:`(*)`
        - Output: :math:`(*)`, where :math:`*` can be any additional number of
          dimensions.

    Example:
        >>> a = np.array([1., 2., 3., 4.])
        >>> spect = Spectrogram(window_size=2, hop_size=2)
        >>> t = Log()
        >>> print(t)
        Log(magnitude_boost=1e-07)
        >>> np.set_printoptions(precision=5)
        >>> t(spect(a))
        array([[1.00000e-07, 1.09861e+00, 1.09861e+00],
               [1.00000e-07, 1.09861e+00, 1.09861e+00]])
    """

    def __init__(
            self,
            *,
            magnitude_boost=1e-7,
    ):
        self.magnitude_boost = magnitude_boost

        if self.magnitude_boost < 0:
            raise ValueError('`magnitude_boost` has to be >=0, but is {}'
                             .format(self.magnitude_boost))

    def __call__(self, signal):
        signal = np.log(signal + self.magnitude_boost)
        return signal

    def __repr__(self):
        options = ('magnitude_boost={}'.format(self.magnitude_boost))
        return '{0}({1})'.format(self.__class__.__name__, options)


class RandomAdditiveMix(object):
    r"""Mix two signals additively by a randomly picked ratio.

    Randomly pick a signal from an augmentation data set and mix it with the
    actual signal by a signal-to-noise ratio in dB randomly selected from a
    list of possible ratios.

    The signal from the augmentation data set is expanded, cropped,
    or has its number of channels adjusted by a downmix or upmix using
    :class:`Remix` if necessary.

    The signal can be expanded by:

        * ``'multiple'`` loading multiple files from the
          augmentation data set and concatenating them along the time axis
        * ``'pad'`` expand the signal by adding trailing zeros
        * ``'replicate'`` replicate the signal to match the specified size.
          If result exceeds specified size after replication, the signal will
          then be cropped

    The signal can be cropped by:

        * ``'start'`` crop signal from the beginning of the file all
          the way to the necessary length
        * ``'random'`` starts at a random offset from the beginning of the file

    * :attr:`dataset` controls the data set used for augmentation
    * :attr:`ratio` controls the ratio in dB between mixed signals
    * :attr:`ratios` controls the ratios to be randomly picked from
    * :attr:`normalize` controls if the mixed signal is normalized
    * :attr:`expand_method` controls if the signal from the augmented data
      set is automatically expanded according to an expansion rule.
      Default: `pad`
    * :attr:`crop_method` controls how the signal is cropped. Is only
      relevant if the augmentation signal is longer than the input one,
      or if `expand_method` is set to `multiple`. Default: `random`
    * :attr:`percentage_silence` controls the percentage of the input data
      that will be mixed with silence. Should be between `0` and `1`.
      Default: `1`
    * :attr:`time_axis` controls time axis for automatic signal adjustment
    * :attr:`channel_axis` controls channel axis for automatic signal
      adjustment
    * :attr:`fix_randomization` controls the randomness of the ratio selection

    Note:
        :attr:`fix_randomization` covers only the selection of the ratio. The
        selection of a signal from the augmentation data set and its signal
        length adjustment will always be random.

    Args:
        dataset (torch.utils.data.Dataset): data set for augmentation
        ratios (list of int, optional): mix ratios in dB to randomly pick from
            (e.g. SNRs). Default: `[0, 15, 30]`
        normalize (bool, optional): normalize mixture. Default: `False`
        expand_method (str, optional): controls the adjustment of
            the length data set that is added to the original data set.
            Default: `pad`
        crop_method (str, optional): controls the crop transform that will be
            called on the mix signal if it is longer than the input signal.
            Default: `random`
        percentage_silence (float, optional): controls the percentage of
            input data that should be augmented with silence. Default: `0`
        time_axis (int, optional): length axis of both data sets. Default: `-1`
        channel_axis (int, optional): channels axis of both data sets.
            Default: `-2`
        fix_randomization (bool, optional): freeze random selection between
            different calls of transform. Default: `False`

    Shape:
        - Input: :math:`(*, C, N, *)`
        - Output: :math:`(*, C, N, *)`, where :math:`C` is the number of
          channels and :math:`N` is the number of samples. They don't have to
          be placed in the order shown here, but the order is preserved during
          transformation.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> from audtorch import datasets
        >>> np.random.seed(0)
        >>> a = np.array([[1, 2], [3, 4]])
        >>> noise = datasets.WhiteNoise(duration=1, sampling_rate=2)
        >>> t = RandomAdditiveMix(noise, ratios=[3], expand_method='pad')
        >>> print(t)
        RandomAdditiveMix(dataset=WhiteNoise, ratios=[3], ratio=None, percentage_silence=0, expand_method=pad, crop_method=random, time_axis=-1, channel_axis=-2)
        >>> np.set_printoptions(precision=8)
        >>> t(a)
        array([[3.67392992, 2.60655362],
               [5.67392992, 4.60655362]])

    """  # noqa: E501

    def __init__(
            self,
            dataset,
            *,
            ratios=[0, 15, 30],
            normalize=False,
            expand_method='pad',
            crop_method='random',
            percentage_silence=0,
            time_axis=-1,
            channel_axis=-2,
            fix_randomization=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.ratios = ratios
        self.ratio = None
        self.percentage_silence = percentage_silence
        self.normalize = normalize
        self.expand_method = expand_method
        self.crop_method = crop_method
        self.time_axis = time_axis
        self.channel_axis = channel_axis
        self.fix_randomization = fix_randomization

        self._remix = Remix(0, axis=channel_axis)
        if self.expand_method != 'multiple':
            self._expand = Expand(0, axis=self.time_axis,
                                  method=self.expand_method)

    def __call__(self, signal1):
        if not self.fix_randomization or not self.ratio:
            self.ratio = random.choice(self.ratios)

        if random.random() < self.percentage_silence:
            signal2 = np.zeros(signal1.shape)
        else:
            signal2 = random.choice(self.dataset)[0]

            samples_signal1 = signal1.shape[self.time_axis]
            samples_signal2 = signal2.shape[self.time_axis]
            channels_signal1 = np.atleast_2d(signal1).shape[self.channel_axis]
            channels_signal2 = np.atleast_2d(signal2).shape[self.channel_axis]

            # Extend too short signal2
            if samples_signal1 > samples_signal2:
                if self.expand_method == 'multiple':
                    self._remix.channels = channels_signal2
                    while samples_signal2 < samples_signal1:
                        new_signal = random.choice(self.dataset)[0]
                        new_signal = self._remix(new_signal)
                        signal2 = np.concatenate((signal2, new_signal),
                                                 axis=self.time_axis)
                        samples_signal2 = signal2.shape[self.time_axis]
                else:
                    self._expand.size = samples_signal1
                    signal2 = self._expand(signal2)
                    samples_signal2 = signal2.shape[self.time_axis]

            # Crop too long signal2 to match signal1
            if samples_signal1 < samples_signal2:
                if self.crop_method == 'random':
                    idx = RandomCrop.random_index(
                        samples_signal2, samples_signal1
                    )
                elif self.crop_method == 'start':
                    idx = (0, samples_signal1)
                signal2 = F.crop(signal2, idx, axis=self.time_axis)

            # Adjust number of channels of signal2 to signal1
            if channels_signal1 != channels_signal2:
                self._remix.channels = channels_signal1
                signal2 = self._remix(signal2)

        mixture = F.additive_mix(signal1, signal2, self.ratio)

        if self.normalize:
            mixture = F.normalize(mixture)

        return mixture

    def __repr__(self):
        options = ('dataset={0}, ratios={1}, ratio={2}, '
                   'percentage_silence={3}, '
                   'expand_method={4}, crop_method={5}, '
                   'time_axis={6}, channel_axis={7}'
                   .format(self.dataset.__class__.__name__, self.ratios,
                           self.ratio, self.percentage_silence,
                           self.expand_method, self.crop_method,
                           self.time_axis, self.channel_axis))
        if self.normalize:
            options += ', normalize=True'
        if self.fix_randomization:
            options += ', fix_randomization=True'
        return '{0}({1})'.format(self.__class__.__name__, options)


class RandomConvolutionalMix(object):
    r"""Convolve the signal with an augmentation data set.

    Randomly pick an impulse response from an augmentation data set and
    convolve it with the signal. The impulse responses have to be
    one-dimensional.

    * :attr:`dataset` controls the data set used for augmentation
    * :attr:`normalize` controls normalisation of convolved signal
    * :attr:`axis` controls axis of upmix

    Args:
        dataset (torch.utils.data.Dataset): data set for augmentation
        normalize (bool, optional): normalize mixture. Default: `False`
        axis (int, optional): axis of convolution. Default: `-1`

    Shape:
        - Input: :math:`(*, N, *)`
        - Output: :math:`(*, N + M - 1, *)`, where :math:`N` is the number of
          samples of the signal and :math:`M` the number of samples of the
          impulse response.
          :math:`*` can be any additional number of dimensions.

    Example:
        >>> from audtorch import datasets
        >>> np.random.seed(0)
        >>> a = np.array([[1, 2], [3, 4]])
        >>> noise = datasets.WhiteNoise(duration=1, sampling_rate=2, transform=np.squeeze)
        >>> t = RandomConvolutionalMix(noise, normalize=True)
        >>> print(t)
        RandomConvolutionalMix(dataset=WhiteNoise, axis=-1, normalize=True)
        >>> np.set_printoptions(precision=8)
        >>> t(a)
        array([[0.21365151, 0.47576767, 0.09692931],
               [0.64095452, 1.        , 0.19385863]])

    """  # noqa: E501
    def __init__(
            self,
            dataset,
            *,
            normalize=False,
            axis=-1,
    ):
        super().__init__()
        self.dataset = dataset
        self.normalize = normalize
        self.axis = axis

    def __call__(self, signal):
        impulse_response = random.choice(self.dataset)[0]
        mixture = np.apply_along_axis(np.convolve, self.axis, signal,
                                      impulse_response)

        if self.normalize:
            mixture = F.normalize(mixture)

        return mixture

    def __repr__(self):
        options = (
            f'dataset={self.dataset.__class__.__name__}, axis={self.axis}'
        )
        if self.normalize:
            options += ', normalize=True'
        return f'{self.__class__.__name__}({options})'
