import os
from warnings import warn

import pandas as pd
import resampy
from torch.utils.data import Dataset

from .utils import (load, sampling_rate_after_transform)
from .utils import (files_and_labels_from_df)


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


class PandasDataset(AudioDataset):
    r"""Data set from pandas.DataFrame.

    Create a data set by accessing the file locations and corresponding labels
    through a pandas.DataFrame.

    You have to specify which labels of the data set you want as target by the
    names of the corresponding columns in the data frame. If you want to select
    one of those columns the label is returned directly in its corresponding
    data type or you can specify a list of columns and the data set will return
    a dictionary containing the labels.

    The filenames of the corresponding audio files have to be specified with
    relative path from `root` in the data frame in a column with the name
    :attr:`col_filename` which defaults to `filename`.

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set
    * :attr:`column_labels` holds the name of the label columns

    Args:
        root (str): root directory of data set
        df (pandas.DataFrame): data frame with filenames and labels. Relative
            from `root`
        sampling_rate (int): sampling rate in Hz of the data set
        column_labels (str or list of str, optional): name of data frame
            column(s) containing the desired labels. Default: `label`
        column_filename (str, optional): name of column holding the file
            names. Default: `filename`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Example:
        >>> data = datasets.PandasDataset(root='/data',
        ...                               df=dataset_dataframe,
        ...                               sampling_rate=44100,
        ...                               column_labels='age')
        >>> print(data)
        Dataset AudioDataset
            Number of data points: 120
            Root Location: /data
            Sampling Rate: 44100Hz
            Label: age
        >>> sig, target = data[0]
        >>> target
        'age'

    """
    def __init__(self, root, df, sampling_rate, column_labels='label',
                 column_filename='filename', transform=None,
                 target_transform=None, download=False):
        files, labels = \
            files_and_labels_from_df(df, root=root,
                                     column_labels=column_labels,
                                     column_filename=column_filename)
        super().__init__(root, files, targets=labels,
                         sampling_rate=sampling_rate, transform=transform,
                         target_transform=target_transform, download=download)
        self.column_labels = column_labels

    def extra_repr(self):
        fmt_str = '    Labels: {}\n'.format(', '.join(self.column_labels))
        return fmt_str


class CsvDataset(PandasDataset):
    r"""Data set from CSV files.

    Create a data set by reading the file locations and corresponding labels
    from a CSV file.

    You have to specify which labels you want as the target of the data set by
    the names of the corresponding columns in the CSV file. If you want to
    select one of those columns the target is returned directly in its
    corresponding data type or you can specify a list of columns and the data
    set will return a dictionary containing the targets.

    The filenames of the corresponding audio files have to be specified with
    relative path from `root` in the CSV file in a column with the name
    :attr:`col_filename` which defaults to `filename`.

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set
    * :attr:`csv_file` holds the path to the used CSV file

    Args:
        root (str): root directory of data set
        csv_file (str): CSV file with filenames and labels. Relative from
            `root`
        sampling_rate (int): sampling rate in Hz of the data set
        column_labels (str or list of str, optional): name of CSV column(s)
            containing the desired labels. Default: `label`
        column_filename (str, optional): name of CSV column holding the file
            names. Default: `filename`
        sep (str, optional): CSV delimiter. Default: `,`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Example:
        >>> data = datasets.CsvDataset(root='/data',
        ...                            csv_file='train.csv',
        ...                            sampling_rate=44100,
        ...                            column_labels='age')
        >>> print(data)
        Dataset AudioDataset
            Number of data points: 120
            Root Location: /data
            Sampling Rate: 44100Hz
            Label: age
            CSV file: train.csv
        >>> sig, target = data[0]
        >>> target
        'age'

    """
    def __init__(self, root, csv_file, sampling_rate, sep=',',
                 column_labels='label', column_filename='filename',
                 transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.csv_file = os.path.join(self.root, csv_file)

        if download:
            self._download()

        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError('CSV file {} not found.'
                                    .format(self.csv_file))
        df = pd.read_csv(self.csv_file, sep)
        super().__init__(root, df, sampling_rate,
                         column_labels=column_labels,
                         column_filename=column_filename, transform=transform,
                         target_transform=target_transform)

    def extra_repr(self):
        fmt_str = super().extra_repr()
        fmt_str += ('    CSV file: {}\n'
                    .format(os.path.basename(self.csv_file)))
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
