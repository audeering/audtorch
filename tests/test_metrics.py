import pytest
import scipy
import torch

import numpy as np

import audtorch.metrics as metrics
import audtorch.metrics.functional as F


@pytest.mark.parametrize('reduction', ['none', 'sum', 'mean'])
def test_energypreservingloss(reduction):
    loss = metrics.EnergyConservingLoss(reduction)
    # Random integers as tensors to avoid precision problems with torch.equal
    input = torch.rand((3, 5), requires_grad=True)
    target = torch.rand((3, 5))
    mixture = torch.rand((3, 5))
    noise = mixture - target
    noise_predicted = mixture - input
    expected_output = (torch.abs(input - target)
                       + torch.abs(noise - noise_predicted))
    output = loss(input, target, mixture)
    if reduction == 'none':
        assert torch.equal(output, expected_output)
    elif reduction == 'sum':
        assert torch.equal(output, torch.sum(expected_output))
    elif reduction == 'mean':
        assert torch.equal(output, torch.mean(expected_output))


@pytest.mark.parametrize('shape', [(5,), (5, 3)])
def test_pearsonr(shape):
    input = torch.rand(shape)
    target = torch.rand(shape)

    if len(shape) == 1:
        r = F.pearsonr(input, target)
        assert r.shape[0] == 1
        np.testing.assert_almost_equal(
            r.numpy()[0], scipy.stats.pearsonr(
                input.numpy(), target.numpy())[0], decimal=6)
    else:
        r = F.pearsonr(input, target)
        assert r.shape[0] == shape[0]
        for index, (input_row, target_row) in enumerate(zip(input, target)):
            np.testing.assert_almost_equal(
                r[index].numpy()[0], scipy.stats.pearsonr(
                    input_row.numpy(), target_row.numpy())[0], decimal=6)

        r = F.pearsonr(input, target, batch_first=False)
        assert r.shape[1] == shape[1]
        for index, (input_col, target_col) in enumerate(
                zip(input.transpose(0, 1), target.transpose(0, 1))):
            np.testing.assert_almost_equal(
                r[:, index].numpy()[0], scipy.stats.pearsonr(
                    input_col.numpy(), target_col.numpy())[0], decimal=6)


@pytest.mark.parametrize('shape', [(5,), (5, 3)])
def test_concordance_cc(shape):
    input = torch.rand(shape)
    target = torch.rand(shape)

    def concordance_cc(x, y):
        r = scipy.stats.pearsonr(x, y)[0]
        ccc = 2 * r * x.std() * y.std() / (x.std() * x.std()
                                           + y.std() * y.std()
                                           + (x.mean() - y.mean())
                                           * (x.mean() - y.mean()))
        return ccc

    if len(shape) == 1:
        ccc = F.concordance_cc(input, target)
        assert ccc.shape[0] == 1
        np.testing.assert_almost_equal(
            ccc.numpy()[0], concordance_cc(input.numpy(), target.numpy()),
            decimal=6)
    else:
        ccc = F.concordance_cc(input, target)
        assert ccc.shape[0] == shape[0]
        for index, (input_row, target_row) in enumerate(zip(input, target)):
            np.testing.assert_almost_equal(
                ccc[index].numpy()[0], concordance_cc(
                    input_row.numpy(), target_row.numpy()), decimal=6)

        ccc = F.concordance_cc(input, target, batch_first=False)
        assert ccc.shape[1] == shape[1]
        for index, (input_col, target_col) in enumerate(
                zip(input.transpose(0, 1), target.transpose(0, 1))):
            np.testing.assert_almost_equal(
                ccc[:, index].numpy()[0], concordance_cc(
                    input_col.numpy(), target_col.numpy()), decimal=6)
