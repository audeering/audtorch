from copy import deepcopy
import threading
import queue

from tqdm import tqdm
import numpy as np


def flatten_list(
        nested_list,
):
    """Flatten an arbitrarily nested list.

    Implemented without  recursion to avoid stack overflows.
    Returns a new list, the original list is unchanged.

    Args:
        nested_list (list): nested list

    Returns:
        list: flattened list

    Example:
        >>> flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]])
        [1, 2, 3, 4, 5]
        >>> flatten_list([[1, 2], 3])
        [1, 2, 3]

    """
    def _flat_generator(nested_list):
        while nested_list:
            sublist = nested_list.pop(0)
            if isinstance(sublist, list):
                nested_list = sublist + nested_list
            else:
                yield sublist
    nested_list = deepcopy(nested_list)
    return list(_flat_generator(nested_list))


def to_tuple(
        input,
        *,
        tuple_len=2,
):
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
            raise ValueError(
                f'Input length expected to be {tuple_len} but was {len(input)}'
            )
        else:
            input = tuple(input)
    except TypeError:
        input = tuple([input] * tuple_len)
    return input


def energy(
        signal,
):
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


def power(
        signal,
):
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


def run_worker_threads(
        num_workers,
        task_fun, params,
        *,
        progress_bar=False,
):
    r"""Run parallel tasks using worker threads.

    Args:
        num_workers (int): number of worker threads
        task_fun (Callable): task function with one or more
            parameters, e.g. x, y, z, and optionally returning a value
        params (list of tuples): list of tuples holding parameters
            for each task, e.g. [(x1, y1, z1), (x2, y2, z2), ...]
        progress_bar (bool): show a progress bar. Default: False

    Returns:
        list: result values in order of `params`

    Example:
        >>> power = lambda x, n: x ** n
        >>> params = [(2, n) for n in range(10)]
        >>> run_worker_threads(3, power, params)
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    """
    num_workers = max(0, num_workers)
    n_tasks = len(params)
    results = [None] * n_tasks

    # do not use more workers as needed
    num_workers = n_tasks if num_workers == 0 else min(num_workers, n_tasks)

    # num_workers == 1 -> run sequentially
    if num_workers == 1:
        for index, param in enumerate(params):
            results[index] = task_fun(*param)

    # number_workers > 1 -> parallalize work
    else:

        # define worker thread
        def _worker():
            while True:
                item = q.get()
                if item is None:
                    break
                index, param = item
                results[index] = task_fun(*param)
                q.task_done()

        # create queue, possibly with a progress bar
        if progress_bar:
            class QueueWithProgbar(queue.Queue):
                def __init__(self, n_tasks, maxsize=0):
                    super().__init__(maxsize)
                    self.pbar = tqdm(total=n_tasks)

                def task_done(self):
                    super().task_done()
                    self.pbar.update(1)
            q = QueueWithProgbar(n_tasks)
        else:
            q = queue.Queue()

        # fill queue
        for index, param in enumerate(params):
            q.put((index, param))

        # start workers
        threads = []
        for i in range(num_workers):
            t = threading.Thread(target=_worker)
            t.start()
            threads.append(t)

        # block until all tasks are done
        q.join()

        # stop workers
        for _ in range(num_workers):
            q.put(None)
        for t in threads:
            t.join()

    return results
