import pytest
import numpy as np

import audtorch as at


xfail = pytest.mark.xfail


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
