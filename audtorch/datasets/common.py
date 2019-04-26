import os
from warnings import warn

import resampy
from torch.utils.data import Dataset

from .utils import (load, sampling_rate_after_transform)


class AudioDataset(Dataset):
    r"""Basic audio signal data set.

    This data set can be used if you have a list of files and a list of
    corresponding targets.

    In addition, this class is a great starting point to inherit from if you
    wish to build your own data set.

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set

    Args:
        root (str): root directory of dataset.
        files (list): list of files
        targets (list): list of targets
        sampling_rate (int): sampling rate in Hz of the data set
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Example:
        >>> files = ['speech.wav', 'noise.wav']
        >>> targets = ['speech', 'noise']
        >>> data = datasets.AudioDataset('/data', files, targets, sampling_rate=8000)
        >>> print(data)
        Dataset AudioDataset
            Number of data points: 2
            Root Location: /data
            Sampling Rate: 8000Hz
        >>> sig, target = data[0]
        >>> target
        'speech'

    """  # noqa: E501
    def __init__(self, root, files, targets, sampling_rate, transform=None,
                 target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.files = [os.path.join(self.root, f) for f in files]
        self.targets = targets
        self.original_sampling_rate = sampling_rate
        self.transform = transform
        self.target_transform = target_transform

        assert len(files) == len(targets)

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        signal, signal_sampling_rate = load(self.files[index])
        # Handle empty signals
        if signal.shape[1] == 0:
            warn('Returning previous file.', UserWarning)
            return self.__getitem__(index - 1)
        # Handle different sampling rate
        if signal_sampling_rate != self.original_sampling_rate:
            warn('Resample from {} to {}'
                 .format(signal_sampling_rate, self.original_sampling_rate),
                 UserWarning)
            signal = resampy.resample(signal, signal_sampling_rate,
                                      self.original_sampling_rate, axis=-1)

        target = self.targets[index]

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target

    @property
    def sampling_rate(self):
        return sampling_rate_after_transform(self)

    def _download(self):
        if self._check_exists():
            return
        print('This data set provides no download functionality')

    def _check_exists(self):
        return os.path.exists(self.root)

    def extra_repr(self):
        r"""Set the extra representation of the data set.

        To print customized extra information, you should reimplement
        this method in your own data set. Both single-line and multi-line
        strings are acceptable.

        The extra information will be shown after the sampling rate entry.

        """
        return ''

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        if self.sampling_rate == self.original_sampling_rate:
            fmt_str += '    Sampling Rate: {}Hz\n'.format(self.sampling_rate)
        else:
            fmt_str += ('    Sampling Rate: {}Hz (original: {}Hz)\n'
                        .format(self.sampling_rate,
                                self.original_sampling_rate))
        fmt_str += self.extra_repr()
        if self.transform:
            fmt_str += _include_repr('Transform', self.transform)
        if self.target_transform:
            fmt_str += _include_repr('Target Transform', self.target_transform)
        return fmt_str


def _include_repr(name, obj):
    """Include __repr__ from other object as indented string.

    Args:
        name (str): Name of the object to be documented, e.g. "Transform".
        obj (object with `__repr__`): Object that provides `__repr__` output.

    Results:
        str: Format string of object to include into another `__repr__`.

    Example:
        >>> t = transforms.Pad(2)
        >>> datasets._include_repr('Transform', t)
        '    Transform: Pad(padding=2, value=0, axis=-1)\n'

    """
    part1 = '    {}: '.format(name)
    part2 = obj.__repr__().replace('\n', '\n' + ' ' * len(part1))
    return '{0}{1}\n'.format(part1, part2)
