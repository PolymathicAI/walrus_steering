from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple, cast

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

CONV_FUNCS = {
    1: (nn.Conv1d, F.conv1d),
    2: (nn.Conv2d, F.conv2d),
    3: (nn.Conv3d, F.conv3d),
}

from torch.profiler import record_function

class VstrideEncoder(nn.Module):
    def __init__(
        self,
        kernel_scales_seq: Tuple[Tuple[int, int], ...],
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (4, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 2,
        groups=12,
        variable_downsample: bool = True,
        variable_deterministic_ds: bool = True,
        learned_pad: bool = True,
    ) -> None:
        super().__init__()

        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.learned_pad = learned_pad
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.spatial_dims = spatial_dims
        self.norm_layer1 = nn.GroupNorm
        self.norm_layer2 = nn.GroupNorm
        self.variable_downsample = variable_downsample
        self.kernel_scales_seq = kernel_scales_seq
        self.variable_deterministic_ds = variable_deterministic_ds

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        if self.spatial_dims == 1:
            conv: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = nn.Conv1d
            self.conv_func = F.conv1d
        elif self.spatial_dims == 2:
            conv = nn.Conv2d
            self.conv_func = F.conv2d
        elif self.spatial_dims == 3:
            conv = nn.Conv3d
            self.conv_func = F.conv3d

        # First convolutional layer
        self.proj1 = conv(
            input_dim,
            inner_dim,
            kernel_size=self.base_kernel1,  # type: ignore
            bias=False,
        )

        # Normalization layer after the first convolutional layer
        self.norm1 = self.norm_layer1(groups, inner_dim, affine=True)
        self.act1 = nn.SiLU()

        self.proj2 = conv(
            inner_dim,
            output_dim,
            kernel_size=self.base_kernel2,  # type: ignore
            bias=False,
        )

        # Normalization layer after the second convolutional layer
        self.norm2 = self.norm_layer2(groups, inner_dim, affine=True)
        self.act2 = nn.SiLU()

    def forward(
        self, x: Tensor, bcs=None, metadata=None, **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        embed_kernel = kwargs["random_kernel"]
        # Apply the first convolution with variable stride and pad accordingly
        # to mimic the effect of variable kernel size
        stride1 = tuple([embed_kernel[i][0] for i in range(self.spatial_dims)])
        stride2 = tuple([embed_kernel[i][1] for i in range(self.spatial_dims)])
        # Calculate necessary padding
        # Currently zero padding, may not be the best choice because I think it causes training to be noisy.
        # Implement learned padding scheme in the future.
        if self.learned_pad:
            # if learned pad is true, implemented in patch_jitterers.py
            padding1 = 0
            padding2 = 0
        else:
            padding1 = tuple(
                [
                    int(np.ceil((self.stride1[i] - stride) / 2.0))
                    for i, stride in enumerate(stride1)
                ]
            )  # type: ignore
            padding2 = tuple(
                [
                    int(np.ceil((self.stride2[i] - stride) / 2.0))
                    for i, stride in enumerate(stride2)
                ]
            )  # type: ignore

        padding1 = cast(Tuple[int, ...], padding1)  # type: ignore
        padding2 = cast(Tuple[int, ...], padding2)  # type: ignore
        # TODO: Fix annoying mypy issues, likely by requiring `padding1` to ALWAYS be a tuple of ints

        # Apply the first convolution
        weight1 = self.proj1.weight
        # x is (T, B, C, H, W, D)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # (T B C H W D) -> (TB C H W D)
        x = x.squeeze((-2, -1))  # (TB C H W D) -> (TB C H [W] [D])
        x = self.conv_func(
            x, weight1, bias=self.proj1.bias, stride=stride1, padding=padding1
        )

        x = self.norm1(x)  # Apply normalization
        x = self.act1(x)  # Apply GELU activation

        # Apply the second convolution
        weight2 = self.proj2.weight
        x = self.conv_func(
            x, weight2, bias=self.proj2.bias, stride=stride2, padding=padding2
        )

        x = self.norm2(x)  # Apply normalization
        x = self.act2(x)

        # Try to add back anything squeezed in the beginning
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)

        return x, kwargs


class AdaptiveDVstrideEncoder(nn.Module):
    def __init__(
        self,
        kernel_scales_seq: Tuple[Tuple[int, int], ...],
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (4, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 2,
        groups=12,
        variable_downsample: bool = True,
        variable_deterministic_ds: bool = True,
        learned_pad: bool = True,
        norm_layer: Callable = nn.GroupNorm,
        activation: Callable = nn.SiLU,
    ) -> None:
        super().__init__()

        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.learned_pad = learned_pad
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.spatial_dims = spatial_dims
        self.variable_downsample = variable_downsample
        self.kernel_scales_seq = kernel_scales_seq
        self.variable_deterministic_ds = variable_deterministic_ds

        conv_class, self.conv_func = CONV_FUNCS[spatial_dims]

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        # First convolutional layer
        self.proj1 = conv_class(
            input_dim,
            inner_dim,
            kernel_size=self.base_kernel1,  # type: ignore
            bias=False,
        )

        # Normalization layer after the first convolutional layer
        self.norm1 = norm_layer(groups, inner_dim, affine=True)
        self.act = activation()

        self.proj2 = conv_class(
            inner_dim,
            output_dim,
            kernel_size=self.base_kernel2,  # type: ignore
            bias=False,
        )

        # Normalization layer after the second convolutional layer
        self.norm2 = norm_layer(groups, output_dim, affine=True)

    def adaptive_conv(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        stride: Sequence[int],
        padding: Sequence[int],
    ) -> Tensor:
        spatial_dims = x.shape[-self.spatial_dims :]
        stride = list(stride)
        padding = list(padding)
        # Dims ordered backward since spatial D is backward
        for i, dim in enumerate(spatial_dims[::-1], start=1):
            if dim == 1:
                weight = weight.sum(dim=-i, keepdim=True)
                stride[-i] = 1
                padding[-i] = 0
        out = self.conv_func(x, weight, bias, tuple(stride), tuple(padding))
        return out

    def forward(
        self, x: Tensor, bcs=None, metadata=None, **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        with record_function("VstrideEncoder.forward"):
            embed_kernel = kwargs["random_kernel"]
            # Apply the first convolution with variable stride and pad accordingly
            # to mimic the effect of variable kernel size
            stride1 = tuple([embed_kernel[i][0] for i in range(self.spatial_dims)])
            stride2 = tuple([embed_kernel[i][1] for i in range(self.spatial_dims)])
            # Calculate necessary padding
            # Currently zero padding, may not be the best choice because I think it causes training to be noisy.
            # Implement learned padding scheme in the future.
            if self.learned_pad:
                # if learned pad is true, implemented in patch_jitterers.py
                padding1 = (0,) * self.spatial_dims
                padding2 = (0,) * self.spatial_dims
            else:
                padding1 = tuple(
                    [
                        int(np.ceil((self.stride1[i] - stride) / 2.0))
                        for i, stride in enumerate(stride1)
                    ]
                )  # type: ignore
                padding2 = tuple(
                    [
                        int(np.ceil((self.stride2[i] - stride) / 2.0))
                        for i, stride in enumerate(stride2)
                    ]
                )  # type: ignore

            padding1 = cast(Tuple[int, ...], padding1)  # type: ignore
            padding2 = cast(Tuple[int, ...], padding2)  # type: ignore

            # Apply the first convolution
            weight1 = self.proj1.weight
            # x is (T, B, C, H, W, D)
            T = x.shape[0]
            # Flatten time
            x = rearrange(x, "T B ... -> (T B) ...")  # (T B C H W D) -> (TB C H W D)
            x = self.adaptive_conv(
                x, weight1, bias=self.proj1.bias, stride=stride1, padding=padding1
            )

            x = self.act(self.norm1(x))

            # Apply the second convolution
            weight2 = self.proj2.weight
            x = self.adaptive_conv(
                x, weight2, bias=self.proj2.bias, stride=stride2, padding=padding2
            )

            x = self.act(self.norm2(x))

            # Try to add back anything squeezed in the beginning
            x = rearrange(x, "(T B) ... -> T B ...", T=T)

            return x, kwargs
