import torch.nn as nn
import torch.nn.functional as F


class EnergyConservingLoss(nn.L1Loss):
    r"""Energy conserving loss.

    A two term loss that enforces energy conservation after
    :cite:`Rethage2018`.

    The loss can be described as:

    .. math::
        \ell(x, y, m) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = |x_n - y_n| + |b_n - \hat{b_n}|,

    where :math:`N` is the batch size. If reduction is not ``'none'``, then:

    .. math::
        \ell(x, y, m) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` is the input signal (estimated target), :math:`y` the target
    signal, :math:`m` the mixture signal, :math:`b` the background signal given
    by :math:`b = m - y`, and :math:`\hat{b}` the estimated background signal
    given by :math:`\hat{b} = m - x`.

    Args:
        reduction (string, optional): specifies the reduction to apply to the
            output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the sum of the output
            will be divided by the number of elements in the output, 'sum': the
            output will be summed.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Mixture: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduction is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> loss = EnergyConservingLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> mixture = torch.randn(3, 5)
        >>> loss(input, target, mixture)
        tensor(2.1352, grad_fn=<AddBackward0>)

    """
    def __init__(self, reduction='mean'):
        super().__init__(None, None, reduction)

    def forward(self, y_predicted, y, x):
        noise = x - y
        noise_predicted = x - y_predicted
        return (F.l1_loss(y_predicted, y, reduction=self.reduction)
                + F.l1_loss(noise_predicted, noise, reduction=self.reduction))
