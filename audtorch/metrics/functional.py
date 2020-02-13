

def pearsonr(
        x,
        y,
        batch_first=True,
):
    r"""Computes Pearson Correlation Coefficient across rows.

    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:

    .. math::

        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}

    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.

    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`

    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`

    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:

        .. math::

            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2

        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.

    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])

    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr


def concordance_cc(
        x,
        y,
        batch_first=True,
):
    r"""Computes Concordance Correlation Coefficient across rows.

    Concordance Correlation Coefficient is computed as:

    .. math::

        \rho_c = \frac {2\rho\sigma_X\sigma_Y} {\sigma_X\sigma_X +
        \sigma_Y\sigma_Y + (\mu_X - \mu_Y)^2}

    where :math:`\rho` is Pearson Correlation Coefficient, :math:`\sigma_X`,
    :math:`\sigma_Y` are the standard deviation, and :math:`\mu_X`,
    :math:`\mu_Y` the mean values of :math:`X` and :math:`Y` accordingly.

    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the concordance correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.

    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.

    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:

        .. math::

            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2

        We therefore account for this correction in the computation of the
        concordance correlation coefficient by multiplying all standard
        deviations with :math:`\frac{N-1}{N}`. This is equivalent to
        multiplying only :math:`(\mu_X - \mu_Y)^2` with :math:`\frac{N}{
        N-1}`. We choose that option for numerical stability.

    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`

    Returns:
        torch.Tensor: concordance correlation coefficient between `x` and `y`

    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors

    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = concordance_cc(input, target)
        >>> print('Concordance Correlation between input and target is {0}'.format(output[:, 0]))
        Concordance Correlation between input and target is tensor([ 0.2605, -0.7862,  0.5298])

    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    bessel_correction_term = (x.shape[dim] - 1) / x.shape[dim]

    r = pearsonr(x, y, batch_first)
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)
    ccc = 2 * r * x_std * y_std / (x_std * x_std
                                   + y_std * y_std
                                   + (x_mean - y_mean)
                                   * (x_mean - y_mean)
                                   / bessel_correction_term)
    return ccc
