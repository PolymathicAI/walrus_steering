from __future__ import annotations

from typing import Tuple, Union, cast

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

CONV_FUNCS = {
    1: (nn.ConvTranspose1d, F.conv_transpose1d),
    2: (nn.ConvTranspose2d, F.conv_transpose2d),
    3: (nn.ConvTranspose3d, F.conv_transpose3d),
}


class VstrideDecoder(nn.Module):
    def __init__(
        self,
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 2,
        groups: int = 12,
        learned_pad: bool = True,
    ) -> None:
        super().__init__()

        self.learned_pad = learned_pad
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.spatial_dims = spatial_dims
        self.norm_layer1 = nn.GroupNorm

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        if self.spatial_dims == 1:
            conv: type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d] = (
                nn.ConvTranspose1d
            )
            self.conv_func = F.conv_transpose1d
        elif self.spatial_dims == 2:
            conv = nn.ConvTranspose2d
            self.conv_func = F.conv_transpose2d
        elif self.spatial_dims == 3:
            conv = nn.ConvTranspose3d
            self.conv_func = F.conv_transpose3d

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
        )

    def forward(
        self,
        x: Tensor,
        state_labels,
        stage_info=None,
        metadata=None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        embed_kernel = stage_info["random_kernel"]
        debed_kernel = tuple((b, a) for (a, b) in embed_kernel)

        stride1 = tuple([debed_kernel[i][0] for i in range(self.spatial_dims)])
        stride2 = tuple([debed_kernel[i][1] for i in range(self.spatial_dims)])

        if self.learned_pad:
            # learned padding is taken care of in patch jitterer
            padding1, padding2 = 0, 0
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

        weight1 = self.proj1.weight
        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D
        x = x.squeeze((-2, -1))  # (T B) C H W D -> (T B) C H [W] [D]

        x = self.conv_func(
            x, weight1, bias=self.proj1.bias, stride=stride1, padding=padding1
        )
        x = self.norm1(x)  # Apply normalization
        x = self.act1(x)  # Apply GELU activation

        weight2 = self.proj2.weight
        x = self.conv_func(
            x,
            weight2[:, state_labels],
            bias=self.proj2.bias[state_labels],  # type: ignore
            stride=stride2,
            padding=padding2,
        )

        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)

        return x


class AdaptiveDVstrideDecoder(nn.Module):
    def __init__(
        self,
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (8, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 3,
        groups: int = 12,
        learned_pad: bool = True,
        norm_layer: nn.Module = nn.GroupNorm,
        activation: nn.Module = nn.SiLU,
    ) -> None:
        super().__init__()

        self.learned_pad = learned_pad
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.spatial_dims = spatial_dims

        # First layer
        self.base_kernel1 = tuple(
            [self.base_kernel_size[i][1] for i in range(self.spatial_dims)]
        )
        self.stride1 = self.base_kernel1

        # Second layer
        self.base_kernel2 = tuple(
            [self.base_kernel_size[i][0] for i in range(self.spatial_dims)]
        )
        self.stride2 = self.base_kernel2

        conv_class, self.conv_func = CONV_FUNCS[self.spatial_dims]

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
        )

    def adaptive_conv_transpose(self, x, weight, bias, stride, padding):
        spatial_dims = x.shape[-self.spatial_dims :]
        stride = list(stride)
        padding = list(padding)
        for i, dim in enumerate(spatial_dims[::-1], start=1):
            if dim == 1:
                weight = weight.mean(dim=-i, keepdim=True)
                stride[-i] = 1
                padding[-i] = 0
        x = self.conv_func(x, weight, bias, tuple(stride), tuple(padding))

        return x

    def forward(
        self,
        x: Tensor,
        state_labels,
        stage_info=None,
        metadata=None,
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        embed_kernel = stage_info["random_kernel"]
        debed_kernel = tuple((b, a) for (a, b) in embed_kernel)

        stride1 = tuple([debed_kernel[i][0] for i in range(self.spatial_dims)])
        stride2 = tuple([debed_kernel[i][1] for i in range(self.spatial_dims)])

        if self.learned_pad:
            # learned padding is taken care of in patch jitterer
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

        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D

        x = self.adaptive_conv_transpose(
            x, self.proj1.weight, bias=self.proj1.bias, stride=stride1, padding=padding1
        )
        x = self.act(self.norm1(x))  # Apply normalization

        x = self.adaptive_conv_transpose(
            x,
            self.proj2.weight[:, state_labels],
            bias=self.proj2.bias[state_labels],  # type: ignore
            stride=stride2,
            padding=padding2,
        )
        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)

        return x
