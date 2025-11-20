from __future__ import annotations

from typing import Any, Dict, Tuple, cast

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from walrus_steering.models.shared_utils.flexi_utils import (
    InterpolationType,
    _cache_pinvs,
    generate_two_conv_combinations,
    resize_patch_embed,
)


class FlexiEncoder(nn.Module):
    def __init__(
        self,
        kernel_scales_seq: Tuple[Tuple[int, int], ...],
        kernel_scales_seq_deterministic: Tuple[Tuple[int, int], ...],
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
        variable_downsample: bool = True,
        variable_deterministic_ds: bool = True,
    ) -> None:
        super().__init__()

        self.base_kernel_size = (
            base_kernel_size2d if spatial_dims == 2 else base_kernel_size3d
        )
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.antialias = antialias
        self.spatial_dims = spatial_dims
        self.interpolation = interpolation
        self.norm_layer = nn.GroupNorm
        self.variable_downsample = variable_downsample
        self.kernel_scales_seq = kernel_scales_seq
        self.kernel_scales_seq_deterministic = kernel_scales_seq_deterministic
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

        if spatial_dims == 1:
            conv: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = nn.Conv1d
            self.conv_func = F.conv1d
            self.interpolation = "nearest"
        elif spatial_dims == 2:
            conv = nn.Conv2d
            self.conv_func = F.conv2d
            self.anti_alias = True
        elif spatial_dims == 3:
            conv = nn.Conv3d
            self.conv_func = F.conv3d
            self.interpolation = "trilinear"

        # First convolutional layer
        self.proj1 = conv(
            input_dim,
            inner_dim,
            kernel_size=self.base_kernel1,  # type: ignore
            bias=False,
        )

        # Normalization layer after the first convolutional layer
        self.norm1 = self.norm_layer(groups, inner_dim, affine=True)
        self.act1 = nn.GELU()

        # Second convolutional layer
        self.proj2 = conv(
            inner_dim,
            output_dim,
            kernel_size=self.base_kernel2,  # type: ignore
            bias=False,
        )

        # Normalization layer after the second convolutional layer
        self.norm2 = self.norm_layer(groups, output_dim, affine=True)

        if not self.variable_deterministic_ds:
            self.kernel_scales_seq1, self.kernel_scales_seq2 = (
                generate_two_conv_combinations(
                    self.kernel_scales_seq, self.spatial_dims
                )
            )
        else:
            self.kernel_scales_seq1, self.kernel_scales_seq2 = (
                generate_two_conv_combinations(
                    self.kernel_scales_seq_deterministic, self.spatial_dims
                )
            )

        # Pre-calculate pinvs for the first and second layer
        self.pinvs1 = _cache_pinvs(
            self.kernel_scales_seq1,
            interpolation=self.interpolation,
            antialias=self.antialias,
            base_kernel_size=cast(Tuple[int, int], self.base_kernel1),  # TODO
            spatial_dims=spatial_dims,
        )
        self.pinvs2 = _cache_pinvs(
            self.kernel_scales_seq2,
            interpolation=self.interpolation,
            antialias=self.antialias,
            base_kernel_size=cast(Tuple[int, int], self.base_kernel2),
            spatial_dims=spatial_dims,
        )

    def forward(
        self, x: Tensor, bcs=None, metadata=None, **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        embed_kernel = kwargs["random_kernel"]
        new_layer1_kernel = tuple(
            [embed_kernel[i][0] for i in range(self.spatial_dims)]
        )
        new_layer2_kernel = tuple(
            [embed_kernel[i][1] for i in range(self.spatial_dims)]
        )

        if new_layer1_kernel != self.base_kernel1:
            weight1 = resize_patch_embed(
                self.proj1.weight,
                self.base_kernel1,
                new_layer1_kernel,
                self.pinvs1,
                spatial_dims=self.spatial_dims,
            )
        else:
            weight1 = self.proj1.weight

        # x is (T, B, C, H, W, D)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # (T B C H W D) -> (TB C H W D)
        x = x.squeeze((-2, -1))  # (TB C H W D) -> (TB C H [W] [D])

        # Apply the first convolution with resized weights
        x = self.conv_func(x, weight1, bias=self.proj1.bias, stride=new_layer1_kernel)
        x = self.norm1(x)  # Apply normalization
        x = self.act1(x)  # Apply GELU activation

        if new_layer2_kernel != self.base_kernel2:
            weight2 = resize_patch_embed(
                self.proj2.weight,
                self.base_kernel2,
                new_layer2_kernel,
                self.pinvs2,
                spatial_dims=self.spatial_dims,
            )
        else:
            weight2 = self.proj2.weight

        # Apply the second convolution with resized weights
        x = self.conv_func(x, weight2, bias=self.proj2.bias, stride=new_layer2_kernel)
        x = self.norm2(x)  # Apply normalization

        # Try to add back anything squeezed in the beginning
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        out_dict = {
            "random_kernel": embed_kernel,
            "pinvs1": self.pinvs1,
            "pinvs2": self.pinvs2,
        }
        return x, out_dict
