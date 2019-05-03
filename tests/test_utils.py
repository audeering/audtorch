import pytest
import numpy as np

import audtorch as at


xfail = pytest.mark.xfail


@pytest.mark.parametrize('nested_list,expected_list', [
    ([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]], [1, 2, 3, 4, 5]),
    ([[1, 2], 3], [1, 2, 3]),
    ([1, 2, 3], [1, 2, 3]),
])
def test_flatten_list(nested_list, expected_list):
    flattened_list = at.utils.flatten_list(nested_list)
    assert flattened_list == expected_list


@pytest.mark.parametrize('input,tuple_len,expected_output', [
    ('aa', 2, ('a', 'a')),
    (2, 1, (2,)),
    (1, 3, (1, 1, 1)),
    ((1, (1, 2)), 2, (1, (1, 2))),
    ([1, 2], 2, (1, 2)),
    pytest.param([1], 2, [], marks=xfail(raises=ValueError)),
    pytest.param([], 2, [], marks=xfail(raises=ValueError)),
])
def test_to_tuple(input, tuple_len, expected_output):
    output = at.utils.to_tuple(input, tuple_len=tuple_len)
    assert output == expected_output


@pytest.mark.parametrize('input,expected_output', [
    (np.array([[2, 2]]), 8),
])
def test_energy(input, expected_output):
    output = at.utils.energy(input)
    assert output == expected_output


@pytest.mark.parametrize('input,expected_output', [
    (np.array([[2, 2]]), 4),
])
def test_power(input, expected_output):
    output = at.utils.power(input)
    assert output == expected_output


@pytest.mark.parametrize('n_workers,task_fun,params', [
    (3, lambda x, n: x ** n, [(2, n) for n in range(10)]),
])
def test_run_worker_threads(n_workers, task_fun, params):
    list1 = at.utils.run_worker_threads(n_workers, task_fun, params)
    list2 = [task_fun(*p) for p in params]
    assert len(list1) == len(list2) and list1 == list2
