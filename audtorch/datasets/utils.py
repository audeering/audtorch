import os
import subprocess
from warnings import warn

import numpy as np
import audiofile as af


def load(filename, duration=None, offset=0):
    r"""Load audio file.

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


def sampling_rate_after_transform(dataset):
    r"""Sampling rate of data set after all transforms are applied.

    A change of sampling rate by a transform is only recognized, if that
    transform has the attribute :attr:`output_sampling_rate`.

    Args:
        dataset (torch.utils.data.Dataset): data set with `sampling_rate`
            attribute or property

    Returns:
        int: sampling rate in Hz after all transforms are applied

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


def ensure_df_columns_contain(df, labels):
    r"""Raise error if list of labels are not in dataframe columns.

    Args:
        df (pandas.dataframe): data set.
        labels (list of str): labels to be expected in `df.columns`.

    """
    if not set(labels) < set(df.columns):
        raise RuntimeError('Only the following labels are allowed: {}'
                           .format(', '.join(df.columns)))


def ensure_df_not_empty(df, labels=None):
    r"""Raise error if dataframe is empty.

    Args:
        df (pandas.dataframe): data set.
        labels (list of str, optional): list of labels used to shrink data
            set. Default: `None`

    """
    error_message = 'No valid data points found in data set'
    if labels is not None:
        error_message += (' for the selected labels: {}'
                          .format(', '.join(labels)))
    if len(df) == 0:
        raise RuntimeError(error_message)


def files_and_labels_from_df(df, root='.', column_labels='label',
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

    """
    if df is None:
        return [], []
    root = os.path.expanduser(root)
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
