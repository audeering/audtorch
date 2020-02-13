import os

from .utils import (download_url, extract_archive, safe_path)
from .base import CsvDataset


__doctest_skip__ = ['*']


class MozillaCommonVoice(CsvDataset):
    """Mozilla Common Voice speech data set.

    Open and publicly available data set of voices from Mozilla:
    https://voice.mozilla.org/en/datasets

    License: CC-0 (public domain)

    Mozilla Common Voice includes the labels `text`, `up_votes`,
    `down_votes`, `age`, `gender`, `accent`, `duration`. You can select one of
    those labels which is returned as a string by the data set as target or you
    can specify a list of the labels and the data set will return a dictionary
    containing those labels. The default label that is returned is `text`.

    * :attr:`root` holds the data set's location
    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`targets` controls the corresponding targets
    * :attr:`sampling_rate` holds the sampling rate of the returned data
    * :attr:`original_sampling_rate` holds the sampling rate of the audio files
      of the data set

    In addition, the following class attribute is available

    * :attr:`url` holds the download link of the data set

    Args:
        root (str): root directory of data set, where the CSV files are
            located, e.g. `/data/MozillaCommonVoice/cv_corpus_v1`
        csv_file (str, optional): name of a CSV file from the `root`
            folder. No absolute path is possible. You are most probably
            interested in `cv-valid-train.csv`, `cv-valid-dev.csv`, and
            `cv-valid-test.csv`. Default: `cv-valid-train.csv`.
        label_type (str or list of str, optional): one of `text`, `up_votes`,
            `down_votes`, `age`, `gender`, `accent`, `duration`. Or a list of
            any combination of those. Default: `text`
        transform (callable, optional): function/transform applied on the
            signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`
        download (bool, optional): download data set if not present.
            Default: `False`

    Note:
        The Mozilla Common Voice data set is constantly growing. If you
        choose to download it, it will always grep the latest version. If
        you require reproducibility of your results, make sure to store a
        safe snapshot of the version you used.

    Example:
        >>> import sounddevice as sd
        >>> data = MozillaCommonVoice('/data/MozillaCommonVoice/cv_corpus_v1')
        >>> print(data)
        Dataset MozillaCommonVoice
            Number of data points: 195776
            Root Location: /data/MozillaCommonVoice/cv_corpus_v1
            Sampling Rate: 48000Hz
            Labels: text
            CSV file: cv-valid-train.csv
        >>> signal, target = data[0]
        >>> target
        'learn to recognize omens and follow them the old king had said'
        >>> sd.play(signal.transpose(), data.sampling_rate)

    """  # noqa: E501

    url = ('https://common-voice-data-download.s3.amazonaws.com/'
           'cv_corpus_v1.tar.gz')

    def __init__(
            self,
            root,
            *,
            csv_file='cv-valid-train.csv',
            label_type='text',
            transform=None,
            target_transform=None,
            download=False,
    ):

        self.root = safe_path(root)
        csv_file = os.path.join(root, csv_file)

        if download:
            self._download()

        super().__init__(
            csv_file=csv_file,
            sampling_rate=48000,
            root=root,
            sep=',',
            column_labels=label_type,
            column_filename='filename',
            transform=transform,
            target_transform=target_transform,
        )

    def _download(self):
        if self._check_exists():
            return
        download_dir = self.root
        corpus = 'cv_corpus_v1'
        if download_dir.endswith(corpus):
            download_dir = download_dir[:-len(corpus)]
        filename = download_url(self.url, download_dir)
        extract_archive(filename, remove_finished=True)
