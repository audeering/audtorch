import os
import subprocess
from warnings import warn
import urllib
import tarfile

from tqdm import tqdm
import numpy as np
import audiofile as af
from torch.utils.data import Subset

from ..utils import run_worker_threads


__doctest_skip__ = ['load']


def load(filename, *, duration=None, offset=0):
    r"""Load audio file.

    If an error occurrs during loading as the file could not be found,
    is empty, or has the wrong format an empty signal is returned and a warning
    shown.

    Args:
        file (str or int or file-like object): file name of input audio file
        duration (float, optional): return only a specified duration in
            seconds. Default: `None`
        offset (float, optional): start reading at offset in seconds.
            Default: `0`

    Returns:
        tuple:

            * **numpy.ndarray**: two-dimensional array with shape
              `(channels, samples)`
            * **int**: sample rate of the audio file

    Example:
        >>> signal, sampling_rate = load('speech.wav')

    """
    signal = np.array([[]])  # empty signal of shape (1, 0)
    sampling_rate = None
    try:
        signal, sampling_rate = af.read(filename,
                                        duration=duration,
                                        offset=offset,
                                        always_2d=True)
    except ValueError:
        warn('File opening error for: {}'.format(filename), UserWarning)
    except (IOError, FileNotFoundError):
        warn('File does not exist: {}'.format(filename), UserWarning)
    except RuntimeError:
        warn('Runtime error for file: {}'.format(filename), UserWarning)
    except subprocess.CalledProcessError:
        warn('ffmpeg conversion failed for: {}'.format(filename), UserWarning)
    return signal, sampling_rate


def download_url(url, root, *, filename=None, md5=None):
    r"""Download a file from an url to a specified directory.

    Args:
        url (str): URL to download file from
        root (str): directory to place downloaded file in
        filename (str, optional): name to save the file under.
            If `None`, use basename of URL. Default: `None`
        md5 (str, optional): MD5 checksum of the download.
            If None, do not check. Default: `None`

    Returns:
       str: path to downloaded file

    """
    root = safe_path(root)
    if not filename:
        filename = os.path.basename(url)
    filename = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # downloads file
    if not os.path.isfile(filename):
        bar_updater = _gen_bar_updater(tqdm(unit='B', unit_scale=True))
        try:
            print('Downloading ' + url + ' to ' + filename)
            urllib.request.urlretrieve(url, filename, reporthook=bar_updater)
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + filename)
                urllib.request.urlretrieve(url, filename,
                                           reporthook=bar_updater)
    return safe_path(filename)


def download_url_list(urls, root, *, num_workers=0):
    r"""Download files from a list of URLs to a specified directory.

    Args:
        urls (list of str or dict): either list of URLs or dictionary
            with URLs as keys and with either filenames or tuples of
            filename and MD5 checksum as values. Uses basename of URL if
            filename is `None`. Performs no check if MD5 checksum is `None`
        root (str): directory to place downloaded files in
        num_workers (int, optional): number of worker threads
            (0 = len(urls)). Default: `0`

    """
    # always convert to dict
    if type(urls) is list:
        urls = {x: None for x in urls}

    # download file and extract
    def _task(url, filename):
        md5 = None
        if type(filename) is tuple:
            filename, md5 = filename
        return download_url(url, root, filename=filename, md5=md5)

    # start workers
    params = [(url, filename) for url, filename in urls.items()]
    return run_worker_threads(num_workers, _task, params)


def extract_archive(filename, *, out_path=None, remove_finished=False):
    r"""Extract archive.

    Currently `tar.gz` and `tar` archives are supported.

    Args:
        filename (str): path to archive
        out_path (str, optional): extract archive in this folder.
            Default: folder where archive is located in
        remove_finished (bool, optional): if `True` remove archive after
            extraction. Default: `False`

    """
    print('Extracting {}'.format(filename))
    if out_path is None:
        out_path = os.path.dirname(filename)
    if filename.endswith('tar.gz'):
        tar = tarfile.open(filename, 'r:gz')
    elif filename.endswith('tar'):
        tar = tarfile.open(filename, 'r:')
    else:
        raise RuntimeError('Archive format not supported.')
    tar.extractall(path=out_path)
    tar.close()
    if remove_finished:
        os.unlink(filename)


def sampling_rate_after_transform(dataset):
    r"""Sampling rate of data set after all transforms are applied.

    A change of sampling rate by a transform is only recognized, if that
    transform has the attribute :attr:`output_sampling_rate`.

    Args:
        dataset (torch.utils.data.Dataset): data set with `sampling_rate`
            attribute or property

    Returns:
        int: sampling rate in Hz after all transforms are applied

    Example:
        >>> from audtorch import datasets, transforms
        >>> t = transforms.Resample(input_sampling_rate=16000,
        ...                         output_sampling_rate=8000)
        >>> data = datasets.WhiteNoise(sampling_rate=16000, transform=t)
        >>> sampling_rate_after_transform(data)
        8000

    """
    sampling_rate = dataset.original_sampling_rate
    try:
        # List of composed transforms
        transforms = dataset.transform.transforms
    except AttributeError:
        # Single transform
        transforms = [dataset.transform]
    for transform in transforms:
        if hasattr(transform, 'output_sampling_rate'):
            sampling_rate = transform.output_sampling_rate
    return sampling_rate


def ensure_same_sampling_rate(datasets):
    r"""Raise error if provided data set differ in sampling rate.

    All data sets that are checked need to have a `sampling_rate` attribute or
    property.

    Args:
        datasets (list of torch.utils.data.Dataset): list of at least two audio
            data sets.

    """
    for dataset in datasets:
        if not hasattr(dataset, 'sampling_rate'):
            raise RuntimeError("{} doesn't have a `sampling_rate` attribute."
                               .format(dataset))
    for n in range(1, len(datasets)):
        if datasets[0].sampling_rate != datasets[n].sampling_rate:
            error_msg = 'Sampling rates do not match:\n'
            for dataset in datasets:
                info = dataset.__repr__()
                error_msg += '{}Hz from {}'.format(dataset.sampling_rate, info)
            raise ValueError(error_msg)


def ensure_df_columns_contain(df, labels):
    r"""Raise error if list of labels are not in dataframe columns.

    Args:
        df (pandas.dataframe): data frame
        labels (list of str): labels to be expected in `df.columns`

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(data=[(1, 2)], columns=['a', 'b'])
        >>> ensure_df_columns_contain(df, ['a', 'c'])
        Traceback (most recent call last):
        RuntimeError: Dataframe contains only these columns: 'a, b'

    """
    ensure_df_not_empty(df)
    if not set(labels) <= set(df.columns):
        raise RuntimeError("Dataframe contains only these columns: '{}'"
                           .format(', '.join(df.columns)))


def ensure_df_not_empty(df, labels=None):
    r"""Raise error if dataframe is empty.

    Args:
        df (pandas.dataframe): data frame
        labels (list of str, optional): list of labels used to shrink data
            set. Default: `None`

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame()
        >>> ensure_df_not_empty(df)
        Traceback (most recent call last):
        RuntimeError: No valid data points found in data set

    """
    error_message = 'No valid data points found in data set'
    if labels is not None:
        error_message += (' for the selected labels: {}'
                          .format(', '.join(labels)))
    if len(df) == 0:
        raise RuntimeError(error_message)


def files_and_labels_from_df(df, *, root='.', column_labels='label',
                             column_filename='filename'):
    r"""Extract list of files and labels from dataframe columns.

    Args:
        df (pandas.DataFrame): data frame with filenames and labels. Relative
            from `root`
        root (str, optional): root directory of data set. Default: `.`
        column_labels (str or list of str, optional): name of data frame
            column(s) containing the desired labels. Default: `label`
        column_filename (str, optional): name of column holding the file
            names. Default: `filename`

    Returns:
        tuple:
            * list of str: list of files
            * list of str or list of dicts: list of labels

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(data=[('speech.wav', 'speech')],
        ...                   columns=['filename', 'label'])
        >>> files, labels = files_and_labels_from_df(df)
        >>> os.path.relpath(files[0]), labels[0]
        ('speech.wav', 'speech')

    """
    if df is None:
        return [], []
    root = safe_path(root)
    if isinstance(column_labels, str):
        column_labels = [column_labels]
    ensure_df_columns_contain(df, column_labels)
    df = df[[column_filename] + column_labels]
    # Drop empty entries
    df = df.dropna().reset_index(drop=True)
    ensure_df_not_empty(df, column_labels)
    # Assign files and labels
    files = df.pop(column_filename).tolist()
    files = [os.path.join(root, f) for f in files]
    if len(column_labels) == 1:
        # list of strings
        labels = df.values.T[0].tolist()
    else:
        # list of dicts
        labels = df.to_dict('records')
    return files, labels


def _gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def defined_split(dataset, split_func):
    r"""Split data set into desired non-overlapping subsets.

    Args:
        dataset (torch.utils.data.Dataset): data set to be split
        split_func (func): function mapping from data set index to subset id,
            :math:`f(\text{index}) = \text{subset\_id}`.
            The target domain of subset ids does not need to cover the
            complete range `[0, 1, ..., (num_subsets - 1)]`

    Returns:
        (list of Subsets): desired subsets according to :attr:`split_func`

    Example:
        >>> import torch
        >>> from torch.utils.data import TensorDataset
        >>> from audtorch.samplers import buckets_of_even_size
        >>> data = TensorDataset(torch.randn(100))
        >>> lengths = np.random.randint(0, 1000, (100,))
        >>> split_func = buckets_of_even_size(lengths, 5)
        >>> subsets = defined_split(data, split_func)
        >>> [len(subset) for subset in subsets]
        [20, 20, 20, 20, 20]

    """
    subset_ids = [split_func(i) for i in range(len(dataset))]
    unique_subset_ids = sorted(set(subset_ids))
    num_subsets = len(unique_subset_ids)

    split_indices = [[] for _ in range(num_subsets)]

    for i, subset_id in enumerate(subset_ids):
        # handle non-coherent target domain
        subset_id = unique_subset_ids.index(subset_id)
        split_indices[subset_id] += [i]

    return [Subset(dataset, indices)
            for indices in split_indices]


def safe_path(path):
    """Ensure the path is absolute and doesn't include `..` or `~`.

    Args:
        path (str): absolute or relative path

    Returns:
        str: absolute path

    """
    return os.path.abspath(os.path.expanduser(path))
