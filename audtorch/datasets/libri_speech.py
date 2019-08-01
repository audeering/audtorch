import os
import glob
import shutil

import pandas as pd

from .utils import (download_url_list, extract_archive, safe_path)
from ..utils import run_worker_threads
from .base import PandasDataset


__doctest_skip__ = ['*']


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
        >>> import sounddevice as sd
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
        >>> sd.play(signal.transpose(), data.sampling_rate)

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

        self.root = safe_path(root)

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
        contents = glob.glob(os.path.join(out_path, 'LibriSpeech/*'))
        for f in contents:
            shutil.move(f, self.root)
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
