import numpy as np


def to_tuple(input, *, tuple_len=2):
    r"""Convert to tuple of given length.

    This utility function is used to convert single-value arguments to tuples
    of appropriate length, e.g. for multi-dimensional inputs where each
    dimension requires the same value. If the argument is already an iterable
    it is returned as a tuple if its length matches the desired tuple length.
    Otherwise a `ValueError` is raised.

    Args:
        input (non-iterable or iterable): argument to be converted to tuple
        tuple_len (int): required length of argument tuple. Default: `2`

    Returns:
        tuple: tuple of desired length

    Example:
        >>> to_tuple(2)
        (2, 2)

    """
    try:
        iter(input)
        if len(input) != tuple_len:
            raise ValueError('Input expected to be of length {} but was of '
                             'length {}'.format(tuple_len, len(input)))
        else:
            input = tuple(input)
    except TypeError:
        input = tuple([input] * tuple_len)
    return input


def energy(signal):
    r"""Energy of input signal.

    .. math::
        E = \sum_n |x_n|^2

    Args:
        signal (numpy.ndarray): signal

    Returns:
        float: energy of signal

    Example:
        >>> a = np.array([[2, 2]])
        >>> energy(a)
        8

    """
    return np.sum(np.abs(signal) ** 2)


def power(signal):
    r"""Power of input signal.

    .. math::
        P = {1 \over N} \sum_n |x_n|^2

    Args:
        signal (numpy.ndarray): signal

    Returns:
        float: power of signal

    Example:
        >>> a = np.array([[2, 2]])
        >>> power(a)
        4.0

    """
    return np.sum(np.abs(signal) ** 2) / signal.size
