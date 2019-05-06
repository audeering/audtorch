from . import functional as F


class PearsonR(object):
    r"""Computes Pearson Correlation Coefficient.

    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:

    .. math::

        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}

    If inputs are vectors, computes Pearson's :math:`\rho` between the two
    of them. If inputs are multi-dimensional arrays, computes Pearson's
    :math:`\rho` along the first or last input dimension according to the
    `batch_first` argument, returns a **torch.Tensor** as output,
    and optionally reduces it according to the `reduction` argument.

    Args:
        reduction (string, optional): specifies the reduction to apply to the
            output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the sum of the output
            will be divided by the number of elements in the output, 'sum': the
            output will be summed. Default: 'mean'
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduction is ``'none'``, then :math:`(N, 1)`

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> metric = PearsonR()
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> metric(input, target)
        tensor(0.1220)

    """
    def __init__(self, reduction='mean', batch_first=True):
        self.reduction = reduction
        self.batch_first = batch_first

    def __call__(self, x, y):
        r = F.pearsonr(x, y, self.batch_first)
        if self.reduction == 'mean':
            r = r.mean()
        elif self.reduction == 'sum':
            r = r.sum()
        return r


class ConcordanceCC(object):
    r"""Computes Concordance Correlation Coefficient (CCC).

    CCC is computed as:

    .. math::

        \rho_c = \frac {2\rho\sigma_X\sigma_Y} {\sigma_X\sigma_X +
        \sigma_Y\sigma_Y + (\mu_X - \mu_Y)^2}

    where :math:`\rho` is Pearson Correlation Coefficient, :math:`\sigma_X`,
    :math:`\sigma_Y` are the standard deviation, and :math:`\mu_X`,
    :math:`\mu_Y` the mean values of :math:`X` and :math:`Y` accordingly.

    If inputs are vectors, computes CCC between the two of them. If inputs
    are multi-dimensional arrays, computes CCC along the first or last input
    dimension according to the `batch_first` argument, returns a
    **torch.Tensor** as output, and optionally reduces it according to the
    `reduction` argument.

    Args:
        reduction (string, optional): specifies the reduction to apply to the
            output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the sum of the output
            will be divided by the number of elements in the output, 'sum': the
            output will be summed. Default: 'mean'
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduction is ``'none'``, then :math:`(N, 1)`

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> metric = ConcordanceCC()
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> metric(input, target)
        tensor(0.0014)

    """
    def __init__(self, reduction='mean', batch_first=True):
        self.reduction = reduction
        self.batch_first = batch_first

    def __call__(self, x, y):
        r = F.concordance_cc(x, y, self.batch_first)
        if self.reduction == 'mean':
            r = r.mean()
        elif self.reduction == 'sum':
            r = r.sum()
        return r
