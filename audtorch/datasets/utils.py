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
    transform has the attribute :attr:`output_sample_rate`.

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
        if hasattr(transform, 'output_sample_rate'):
            sampling_rate = transform.output_sample_rate
    return sampling_rate
