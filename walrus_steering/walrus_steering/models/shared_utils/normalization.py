import torch
import torch.nn as nn
import torch.nn.functional as F


def get_spatial_dims(n_dims: int, include_time: bool):
    """Assumes input is ([T], B, H, [W, D], C)"""
    start = 1
    if include_time:
        start += 1
    return list(range(start, start + n_dims))


# class RMSGroupNorm(nn.Module):
#     def __init__(self, heads, dim):
#         super().__init__()
#         self.scale = dim ** 0.5
#         self.gamma = nn.Parameter(torch.ones(heads, 1, dim) / self.scale)
#         self.heads = heads

#     def forward(self, x, n_dims, include_time=False):
#         # Assume input is ([T], B, H, [W, D], C)
#         spatial_dims = get_spatial_dims(n_dims, include_time)
#         x = x.view(*x.shape[:-1], self.heads,-1)
#         x = x.permute(*x.shape[:spatial_dims[0]], -2, *spatial_dims, -1)
#         normed = F.normalize(x, dim = -1)
#         return normed * self.scale * self.gamma


class RMSGroupNorm(nn.Module):
    r"""Applies RMS version of Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Assume input is (B, C, H, W, D)
        dims = list(input.shape[2:])
        input = input.view(input.shape[0], self.num_groups, -1, *dims)
        norm_shape = input.shape[3:]
        input = F.rms_norm(input, normalized_shape=norm_shape)
        input = input.view(input.shape[0], -1, *dims)
        if self.weight is not None:
            indexing_tuple = (slice(None),) + (None,) * len(dims)
            return input * self.weight[indexing_tuple]
        else:
            return input

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, " "affine={affine}".format(
            **self.__dict__
        )
