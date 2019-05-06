import os
import glob
import pandas as pd

from .utils import (download_url, download_url_list, extract_archive)
from ..utils import run_worker_threads
from .common import CsvDataset
from .common import PandasDataset


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

    def __init__(self, root, *, csv_file='cv-valid-train.csv',
                 label_type='text', transform=None, target_transform=None,
                 download=False):

        if download:
            self.root = os.path.expanduser(root)
            self._download()

        super().__init__(root, csv_file, sampling_rate=48000, sep=',',
                         column_labels=label_type, column_filename='filename',
                         transform=transform,
                         target_transform=target_transform)

    def _download(self):
        if self._check_exists():
            return
        download_dir = self.root
        corpus = 'cv_corpus_v1'
        if download_dir.endswith(corpus):
            download_dir = download_dir[:-len(corpus)]
        filename = download_url(self.url, download_dir)
        extract_archive(filename, remove_finished=True)


class LibriSpeech(PandasDataset):
    r"""`LibriSpeech` speech data set.

    Open and publicly available data set of voices from OpenSLR:
    http://www.openslr.org/12/

    License: CC BY 4.0.

    `LibriSpeech` contains several hundred hours of English speech
    with corresponding transcriptions in capital letters without punctuation.

    It is split into different subsets according to WER-level achieved when
    performing speech recognition on the speakers. The subsets are:
    `train-clean-100`, `train-clean-360`, `train-other-500` `dev-clean`,
    `dev-other`, `test-clean`, `test-other`

    * :attr:`root` holds the data set's location
    * :attr:`transform` controls the input transform
    * :attr:`target_transform` controls the target transform
    * :attr:`files` controls the audio files of the data set
    * :attr:`labels` controls the corresponding labels
    * :attr:`sampling_rate` holds the sampling rate of data set

    In addition, the following class attributes are available

    * :attr:`all_sets` holds the names of the different pre-defined sets
    * :attr:`urls` holds the download links of the different sets

    Args:
        root (str): root directory of data set
        sets (str or list, optional): desired sets of `LibriSpeech`.
            Mutually exclusive with :attr:`dataframe`.
            Default: `None`
        dataframe (pandas.DataFrame, optional): pandas data frame containing
            columns `audio_path` (relative to root) and `transcription`.
            It can be used to pre-select files based on meta information,
            e.g. sequence length. Mutually exclusive with :attr:`sets`.
            Default: `None`
        transform (callable, optional): function/transform applied on
            the signal. Default: `None`
        target_transform (callable, optional): function/transform applied on
            the target. Default: `None`
        download (bool, optional): download data set to root directory
            if not present. Default: `False`

    Example:
        >>> data = LibriSpeech(root='/data/LibriSpeech', sets='dev-clean')
        >>> print(data)
        Dataset LibriSpeech
            Number of data points: 2703
            Root Location: /data/LibriSpeech
            Sampling Rate: 16000Hz
            Sets: dev-clean
        >>> signal, label = data[8]
        >>> label
        AS FOR ETCHINGS THEY ARE OF TWO KINDS BRITISH AND FOREIGN
        >>> sounddevice.play(signal.transpose(), data.sampling_rate)

    """

    all_sets = ['train-clean-100', 'train-clean-360', 'train-other-500',
                'dev-clean', 'dev-other', 'test-clean', 'test-other']
    urls = {
        "train-clean-100":
            'https://openslr.org/resources/12/train-clean-100.tar.gz',
        "train-clean-360":
            'https://openslr.org/resources/12/train-clean-360.tar.gz',
        "train-other-500":
            'https://openslr.org/resources/12/train-other-500.tar.gz',
        "dev-clean":
            'https://openslr.org/resources/12/dev-clean.tar.gz',
        "dev-other":
            'https://openslr.org/resources/12/dev-other.tar.gz',
        "test-clean":
            'https://openslr.org/resources/12/test-clean.tar.gz',
        "test-other":
            'https://openslr.org/resources/12/test-other.tar.gz'}
    _transcription = 'transcription'
    _audio_path = 'audio_path'

    def __init__(self, root, *, sets=None, dataframe=None,
                 transform=None, target_transform=None, download=False):

        self.root = os.path.expanduser(root)

        if isinstance(sets, str):
            sets = [sets]
        if dataframe is None and sets is None:
            self.sets = self.all_sets
        elif dataframe is None:
            assert set(sets) <= set(self.all_sets)
            self.sets = sets
        elif dataframe is not None:
            self.sets = None
        else:
            raise ValueError('Either `sets` or `dataframe` can be specified.')

        if download:  # data not available
            self._download()

        if not self._check_exists():
            raise RuntimeError('Requested sets of data set not found.')

        if dataframe is None:
            files = self._get_files()
            dataframe = self._create_dataframe(files)

        super().__init__(
            root=self.root,
            sampling_rate=16000,
            df=dataframe,
            column_filename=self._audio_path,
            column_labels=self._transcription,
            transform=transform,
            target_transform=target_transform)

    def _check_exists(self):
        return all([os.path.exists(os.path.join(self.root, s))
                    for s in self.sets])

    def _download(self):
        absent_sets = [s for s in self.sets
                       if not os.path.exists(os.path.join(self.root, s))]
        if not absent_sets:
            return

        out_path = os.path.join(self.root, "tmp")
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        urls = [self.urls[s] for s in absent_sets]
        filenames = download_url_list(urls, out_path, num_workers=0)
        for filename in filenames:
            extract_archive(os.path.join(out_path, filename),
                            out_path=out_path,
                            remove_finished=True)
        os.system('mv {} {}'.format(os.path.join(out_path, 'LibriSpeech/*'),
                                    self.root))
        os.rmdir(os.path.join(out_path, "LibriSpeech"))
        os.rmdir(out_path)

    def _get_files(self):
        files = []
        for set in self.sets:
            path_to_files = os.path.join(self.root, set, '**/**/*.flac')
            files += glob.glob(path_to_files)
        return files

    @classmethod
    def _create_dataframe(cls, files):

        def _create_df_per_txt(txt_file):
            _set = txt_file.rsplit('/', 4)[-4]
            df_per_txt = pd.read_csv(txt_file, names=[cls._audio_path])

            # split content once with delimiter
            df_per_txt[[cls._audio_path, cls._transcription]] = \
                df_per_txt[cls._audio_path].str.split(" ", 1, expand=True)

            # compose audio paths relative to root
            df_per_txt[cls._audio_path] = df_per_txt[cls._audio_path].apply(
                _compose_relative_audio_paths(_set))
            return df_per_txt

        def _compose_relative_audio_paths(_set):
            return lambda row: os.path.join(
                _set, *row.split('-')[:-1], row + '.flac')

        # get absolute paths to txt files from audio files
        txt_files = sorted(list(set(
            [f.rsplit('-', 1)[0] + '.trans.txt' for f in files])))
        args = [(f, ) for f in txt_files]
        dataframes = run_worker_threads(12, _create_df_per_txt, args)
        return pd.concat(dataframes, ignore_index=True)

    def extra_repr(self):
        if self.sets is not None:
            fmt_str = '    Sets: {}\n'.format(", ".join(self.sets))
        return fmt_str
