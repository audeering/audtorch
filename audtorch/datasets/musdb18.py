import os
import random

import numpy as np
import musdb
from torch.utils.data import Dataset

from .base import _include_repr
from .utils import sampling_rate_after_transform


__doctest_skip__ = ['*']


class Musdb18(Dataset):
    r"""Musical Source Separation musdb18 data set.

    Data set used in the SiSEC MUS challange 2018:
    https://sigsep.github.io/datasets/musdb.html

    License: restricted to research, varies for different tracks.

    `musdb18` contains 150 full length music tracks (~10 hours) of different
    styles along with their isolated `drums`, `bass`, `vocals`, and `others`
    setms.

    It is split into a `train` subset containing 100 songs, and a `test` subset
    containing the remaining 50 songs. Supervised approaches should be trained
    on the training set and tested on both sets.

    `musdb18` is composed by 100 tracks from the `DSD100 dataset`_, 46 tracks
    from MedleyDB_, 2 tracks from Native Instruments, and 2 tracks from the
    `heise stems remix competition`_.

    .. _DSD100 dataset: https://sigsep.github.io/datasets/dsd100.html
    .. _MedleyDB: http://medleydb.weebly.com/
    .. _Native Instruments: https://www.native-instruments.com/en/specials/stems-for-all/free-stems-tracks/
    .. _heise stems remix competition: https://www.heise.de/ct/artikel/c-t-Remix-Wettbewerb-The-Easton-Ellises-2542427.html#englisch

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`tracks` controls musdb tracks object
    * :attr:`subset` holds the chosen subset
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set

    In addition, the following class attributes are available:

    * :attr:`sets` holds the available subsets

    Args:
        root (str): root directory of dataset.
        subset (str, optional): if not `None` use only a subset of all tracks.
            Can be `train` or `test`. Default: `None`
        percentage_silence (float, optional): value between `0` and `1` which
            controls the random insertion of accompaniment, silence pairs.
            Default: `0`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`
        joint_transform (callable, optional): function/transform applied on the
            signal and target simulaneously. If the transform includes
            randomization it is applied with the same random parameter during
            both calls. Defaut: `None`

    Example:
        >>> data = Musdb18(root='/data/musdb', subset='test')
        >>> print(data)
        Dataset Musdb18
            Number of data points: 50
            Root Location: /data/musdb
            Sampling Rate: 44100Hz
        >>> signal, target = data[0]
        >>> target.shape
        (2, 9256960)

    """  # noqa: E501

    sets = ['train', 'test']

    def __init__(
            self,
            root,
            *,
            subset=None,
            percentage_silence=0,
            transform=None,
            target_transform=None,
            joint_transform=None
    ):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.original_sampling_rate = 44100
        self.subset = subset
        self.percentage_silence = percentage_silence
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform

        if subset is not None and subset not in self.sets:
            raise ValueError('Subset {} not available.'.format(subset))
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        if not (0 <= self.percentage_silence <= 1):
            raise ValueError('`percentage_silence` needs to be in [0, 1]`')

        if self.subset is None:
            self.tracks = musdb.DB(root=self.root)
        else:
            self.tracks = musdb.DB(root=self.root, subsets=self.subset)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        signal = self.tracks[index].audio.T
        target = self.tracks[index].targets['vocals'].audio.T

        # Randomly add (accompaniment, silence) as (signal, target)
        if random.random() < self.percentage_silence:
            signal = self.tracks[index].targets['accompaniment'].audio.T
            target = np.zeros(target.shape)

        if self.joint_transform is not None:
            randomness = getattr(self.joint_transform, 'fix_randomization',
                                 None)
            signal = self.joint_transform(signal)
            if randomness is not None:
                self.joint_transform.fix_randomization = True
            target = self.joint_transform(target)
            if randomness is not None:
                self.joint_transform.fix_randomization = randomness

        if self.transform is not None:
            signal = self.transform(signal)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return signal, target

    @property
    def sampling_rate(self):
        return sampling_rate_after_transform(self)

    def _check_exists(self):
        return all([os.path.exists(os.path.join(self.root, s))
                    for s in self.sets])

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
        if self.subset is not None:
            fmt_str += '    Subset: {}\n'.format(self.subset)
        if self.percentage_silence > 0:
            fmt_str += ('    Silence augmentation: {:.0f}%\n'
                        .format(100 * self.percentage_silence))
        if self.transform:
            fmt_str += _include_repr('Transform', self.transform)
        if self.target_transform:
            fmt_str += _include_repr('Target Transform', self.target_transform)
        if self.joint_transform:
            fmt_str += _include_repr('Joint Transform', self.joint_transform)
        return fmt_str
