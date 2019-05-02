import pytest
import torch
from audtorch import collate


# data format FS = (feature dim, sequence dim)
batch_1 = [[torch.zeros(4, 5), torch.zeros(10)],
           [torch.zeros(4, 6), torch.zeros(12)]]
sequence_dimensions_1 = [-1, -1]
expected_1 = [[2, 4, 6], [2, 6, 4], [6, 2, 4]]

# data format CFS = (channel dim, feature dim, sequence dim)
batch_2 = [[torch.zeros(3, 4, 5), torch.zeros(10)],
           [torch.zeros(3, 4, 6), torch.zeros(12)]]
sequence_dimensions_2 = [2, 0]
expected_2 = [[2, 3, 4, 6], [2, 6, 3, 4], [6, 2, 3, 4]]


@pytest.mark.parametrize("batch,sequence_dimensions,expected",
                         [(batch_1, sequence_dimensions_1, expected_1),
                          (batch_2, sequence_dimensions_2, expected_2)])
@pytest.mark.parametrize("batch_first", [None, True, False])
def test_seq2seq(batch, sequence_dimensions, expected, batch_first):

    collation = collate.Seq2Seq(
        sequence_dimensions=sequence_dimensions,
        batch_first=batch_first,
        pad_values=[-1, -1])
    output = collation(batch)

    if batch_first is None:
        assert list(output[0].shape) == expected[0]

    if batch_first:
        assert list(output[0].shape) == expected[1]

    if batch_first is False:
        assert list(output[0].shape) == expected[2]
