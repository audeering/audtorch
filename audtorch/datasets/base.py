import os
from warnings import warn

import pandas as pd
import resampy
from tabulate import tabulate
from torch.utils.data import (Dataset, ConcatDataset)

from .utils import (
    ensure_df_columns_contain,
    ensure_same_sampling_rate,
    files_and_labels_from_df,
    load,
    safe_path,
    sampling_rate_after_transform,
)


__doctest_skip__ = ['*']


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
    * :attr:`duration` controls audio duration for every file in seconds
    * :attr:`offset` controls audio offset for every file in seconds
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set

    Args:
        files (list): list of files
        targets (list): list of targets
        sampling_rate (int): sampling rate in Hz of the data set
        root (str, optional): root directory of dataset. Default: `None`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Example:
        >>> data = AudioDataset(files=['speech.wav', 'noise.wav'],
        ...                     targets=['speech', 'noise'],
        ...                     sampling_rate=8000,
        ...                     root='/data')
        >>> print(data)
        Dataset AudioDataset
            Number of data points: 2
            Root Location: /data
            Sampling Rate: 8000Hz
        >>> signal, target = data[0]
        >>> target
        'speech'

    """

    def __init__(
            self,
            *,
            files,
            targets,
            sampling_rate,
            root=None,
            transform=None,
            target_transform=None
    ):
        if root is not None:
            self.root = safe_path(root)
            self.files = [os.path.join(self.root, f) for f in files]
        else:
            self.root = root
            self.files = files
        self.targets = targets
        self.original_sampling_rate = sampling_rate
        self.transform = transform
        self.target_transform = target_transform
        # Initialize empty duration and offset attributes.
        self.duration = [None] * self.__len__()
        self.offset = [0] * self.__len__()

        assert len(files) == len(targets)

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        signal, signal_sampling_rate = load(
            self.files[index],
            duration=self.duration[index],
            offset=self.offset[index],
        )
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

    def _check_exists(self):
        if self.root is not None:
            return os.path.exists(self.root)
        else:
            return True

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
        if self.root is not None:
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
    absolute path. If they are relative to a folder, you have to use the
    :obj:`root` argument to specify that folder.

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set
    * :attr:`column_labels` holds the name of the label columns

    Args:
        df (pandas.DataFrame): data frame with filenames and labels
        sampling_rate (int): sampling rate in Hz of the data set
        root (str, optional): root directory added before the files listed
            in the CSV file. Default: `None`
        column_labels (str or list of str, optional): name of data frame
            column(s) containing the desired labels. Default: `label`
        column_filename (str, optional): name of column holding the file
            names. Default: `file`
        column_start (str, optional): name of column holding start of audio
            in the corresponding file in seconds. Default: `None`
        column_end (str, optional): name of column holding end of audio
            in the corresponding file in seconds. Default: `None`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Example:
        >>> data = PandasDataset(root='/data',
        ...                      df=dataset_dataframe,
        ...                      sampling_rate=44100,
        ...                      column_labels='age')
        >>> print(data)
        Dataset AudioDataset
            Number of data points: 120
            Root Location: /data
            Sampling Rate: 44100Hz
            Label: age
        >>> signal, target = data[0]
        >>> target
        31

    """

    def __init__(
            self,
            *,
            df,
            sampling_rate,
            root=None,
            column_labels=None,
            column_filename='file',
            column_start=None,
            column_end=None,
            transform=None,
            target_transform=None
    ):
        files, labels = files_and_labels_from_df(
            df,
            column_labels=column_labels,
            column_filename=column_filename,
        )
        super().__init__(
            files=files,
            targets=labels,
            sampling_rate=sampling_rate,
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        self.column_labels = column_labels
        if column_start is not None:
            ensure_df_columns_contain(df, [column_start])
            self.offset = df[column_start]
        if column_end is not None:
            ensure_df_columns_contain(df, [column_end])
            start = self.offset
            end = df[column_end]
            self.duration = (end - start).where((pd.notnull(end)), None)

    def extra_repr(self):
        if self.column_labels is not None:
            fmt_str = '    Labels: {}\n'.format(self.column_labels)
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
    absolute path. If they are relative to a folder, you have to use the
    :obj:`root` argument to specify that folder.

    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set
    * :attr:`csv_file` holds the path to the used CSV file

    Args:
        csv_file (str): CSV file with filenames and labels
        sampling_rate (int): sampling rate in Hz of the data set
        root (str, optional): root directory added before the files listed
            in the CSV file. Default: `None`
        column_labels (str or list of str, optional): name of CSV column(s)
            containing the desired labels. Default: `label`
        column_filename (str, optional): name of CSV column holding the file
            names. Default: `file`
        column_start (str, optional): name of column holding start of audio
            in the corresponding file in seconds. Default: `None`
        column_end (str, optional): name of column holding end of audio
            in the corresponding file in seconds. Default: `None`
        sep (str, optional): CSV delimiter. Default: `,`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`

    Example:
        >>> data = CsvDataset(csv_file='/data/train.csv',
        ...                   sampling_rate=44100,
        ...                   column_labels='age')
        >>> print(data)
        Dataset AudioDataset
            Number of data points: 120
            Sampling Rate: 44100Hz
            Label: age
            CSV file: /data/train.csv
        >>> signal, target = data[0]
        >>> target
        31

    """

    def __init__(
            self,
            *,
            csv_file,
            sampling_rate,
            root=None,
            sep=',',
            column_labels=None,
            column_filename='file',
            column_start=None,
            column_end=None,
            transform=None,
            target_transform=None
    ):
        self.csv_file = safe_path(csv_file)

        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError('CSV file {} not found.'
                                    .format(self.csv_file))
        df = pd.read_csv(self.csv_file, sep)
        super().__init__(
            df=df,
            sampling_rate=sampling_rate,
            root=root,
            column_labels=column_labels,
            column_filename=column_filename,
            column_start=column_start,
            column_end=column_end,
            transform=transform,
            target_transform=target_transform,
        )

    def extra_repr(self):
        fmt_str = super().extra_repr()
        fmt_str += '    CSV file: {}\n'.format(self.csv_file)
        return fmt_str


class AudioConcatDataset(ConcatDataset):
    r"""Concatenation data set of multiple audio data sets.

    This data set checks that all audio data sets are
    compatible with respect to the sampling rate which they
    are processed with.

    * :attr:`sampling_rate` holds the consistent sampling rate of the
      concatenated data set
    * :attr:`datasets` holds a list of all audio data sets
    * :attr:`cumulative_sizes` holds a list of sizes accumulated over all
      audio data sets, i.e. `[len(data1), len(data1) + len(data2), ...]`

    Args:
        datasets (list of audtorch.AudioDataset): Audio data sets
            with property `sampling_rate`.

    Example:
        >>> import sounddevice as sd
        >>> from audtorch.datasets import LibriSpeech
        >>> dev_clean = LibriSpeech(root='/data/LibriSpeech', sets='dev-clean')
        >>> dev_other = LibriSpeech(root='/data/LibriSpeech', sets='dev-other')
        >>> data = AudioConcatDataset([dev_clean, dev_other])
        >>> print(data)
        Data set AudioConcatDataset
        Number of data points: 5567
        Sampling Rate: 16000Hz
        <BLANKLINE>
        data sets      data points  extra
        -----------  -------------  ---------------
        LibriSpeech           2703  Sets: dev-clean
        LibriSpeech           2864  Sets: dev-other
        >>> signal, label = data[8]
        >>> label
        AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN
        >>> sd.play(signal.transpose(), data.sampling_rate)

    """

    def __init__(self, datasets):
        super().__init__(datasets)
        ensure_same_sampling_rate(datasets)

    @property
    def sampling_rate(self):
        return self.datasets[0].sampling_rate

    def extra_repr(self):
        r"""Set the extra representation of the data set.

        To print customized extra information, you should reimplement
        this method in your own data set. Both single-line and multi-line
        strings are acceptable.

        The extra information will be shown after the sampling rate entry.

        """
        return ''

    def __repr__(self):
        fmt_str = 'Data set ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Sampling Rate: {}Hz\n'.format(self.sampling_rate)
        fmt_str += self.extra_repr()

        headers = ['data sets', 'data points',
                   'transform', 'target_transform', 'extra']
        tabular_data = [(d.__class__.__name__,
                         d.__len__(),
                         d.transform,
                         d.target_transform,
                         d.extra_repr()) for d in self.datasets]

        col = 0  # remove columns without any entries
        for _ in range(len(headers)):
            if not any([True if row[col] else False for row in tabular_data]):
                del headers[col]
                tabular_data = [row[:col] + row[col + 1:]
                                for row in tabular_data]
            else:
                col += 1

        fmt_str += '\n' + tabulate(tabular_data=tabular_data, headers=headers)
        return fmt_str


def _include_repr(name, obj):
    r"""Include __repr__ from other object as indented string.

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
