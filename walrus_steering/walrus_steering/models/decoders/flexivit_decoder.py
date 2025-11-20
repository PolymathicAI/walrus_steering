from __future__ import annotations

from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from walrus_steering.models.shared_utils.flexi_utils import (
    InterpolationType,
    resize_patch_embed,
)


class FlexiDecoder(nn.Module):
    def __init__(
        self,
        base_kernel_size1d: Tuple[Tuple[int, int], ...] = ((4, 4),),
        base_kernel_size2d: Tuple[Tuple[int, int], ...] = ((8, 4), (4, 4)),
        base_kernel_size3d: Tuple[Tuple[int, int], ...] = ((4, 4), (4, 4), (4, 4)),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims: int = 2,
        bias: bool = True,
        antialias: bool = False,
        interpolation: InterpolationType = "bicubic",
        groups: int = 12,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.antialias = antialias
        self.spatial_dims = spatial_dims
        self.interpolation = interpolation
        self.norm_layer = nn.GroupNorm

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

        if spatial_dims == 1:
            conv: type[nn.ConvTranspose1d | nn.ConvTranspose2d | nn.ConvTranspose3d] = (
                nn.ConvTranspose1d
            )
            self.conv_func = F.conv_transpose1d
            self.interpolation = "nearest"
        elif spatial_dims == 2:
            conv = nn.ConvTranspose2d
            self.conv_func = F.conv_transpose2d
            self.antialias = True
        else:
            # anti-aliasing is not supported for 3D.
            conv = nn.ConvTranspose3d
            self.conv_func = F.conv_transpose3d
            self.interpolation = "trilinear"

        self.proj1 = conv(
            input_dim,
            inner_dim,
            kernel_size=self.base_kernel1,  # type: ignore
            bias=False,
        )

        # Normalization layer after the first convolutional layer
        self.norm1 = self.norm_layer(groups, inner_dim, affine=True)
        self.act1 = nn.GELU()

        self.proj2 = conv(
            inner_dim,
            output_dim,
            kernel_size=self.base_kernel2,  # type: ignore
        )

    def forward(
        self, x: Tensor, state_labels, stage_info=None, metadata=None
    ) -> Union[Tensor, Tuple[Tensor, Tuple[int, int]]]:
        if stage_info:
            embed_kernel = stage_info["random_kernel"]
            pinvs1 = stage_info["pinvs1"]
            pinvs2 = stage_info["pinvs2"]

        debed_kernel = tuple((b, a) for (a, b) in embed_kernel)
        new_layer1_kernel = tuple(
            [debed_kernel[i][0] for i in range(self.spatial_dims)]
        )
        new_layer2_kernel = tuple(
            [debed_kernel[i][1] for i in range(self.spatial_dims)]
        )

        if new_layer1_kernel != self.base_kernel1:
            weight1 = resize_patch_embed(
                self.proj1.weight,
                self.base_kernel1,
                new_layer1_kernel,
                pinvs2,
                spatial_dims=self.spatial_dims,
            )
        else:
            weight1 = self.proj1.weight

        # x is (T, B, C, H, W, D)
        # state_labels is (C_in)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # T B C H W D -> (T B) C H W D
        x = x.squeeze((-2, -1))  # (T B) C H W D -> (T B) C H [W] [D]

        # Apply the first convolution with resized weights
        x = self.conv_func(x, weight1, bias=self.proj1.bias, stride=new_layer1_kernel)
        x = self.norm1(x)  # Apply normalization
        x = self.act1(x)  # Apply GELU activation

        if new_layer2_kernel != self.base_kernel2:
            weight2 = resize_patch_embed(
                self.proj2.weight,
                self.base_kernel2,
                new_layer2_kernel,
                pinvs1,
                spatial_dims=self.spatial_dims,
            )
        else:
            weight2 = self.proj2.weight

        # Apply the second convolution with resized weights
        x = self.conv_func(
            x,
            weight2[:, state_labels],
            bias=self.proj2.bias[state_labels],  # type: ignore
            stride=new_layer2_kernel,
        )
        # Do twice for 3d/1d
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        return x
