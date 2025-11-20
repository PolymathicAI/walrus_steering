import itertools
import random
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from functorch import vmap
from torch import Tensor


def choose_kernel_size_random(kernel_scales_seq):
    """
    Choose a kernel size from kernel_scales_seq with uniform probability
    """
    return random.choices(kernel_scales_seq)[0]


def create_patch_dict(kernel_scales_seq: Tuple[Tuple[int, int], ...]):
    patch_sizes = [p[0] * p[1] for p in kernel_scales_seq]
    # Generate a dictionary mapping patch sizes to partitions
    return dict(zip(patch_sizes, kernel_scales_seq))


def generate_patch_combinations(kernel_scales_seq, spatial_dims):
    """
    Generate all possible patch combinations for a given spatial dimension
    """
    patch_sizes = [p[0] * p[1] for p in kernel_scales_seq]
    if spatial_dims == 1:
        patch_combinations = [
            [
                p,
            ]
            for p in patch_sizes
        ]
    elif spatial_dims == 2:
        patch_combinations = list(itertools.product(patch_sizes, repeat=2))
    elif spatial_dims == 3:
        patch_combinations = list(itertools.product(patch_sizes, repeat=3))
    else:
        raise ValueError("Spatial dimension must be 1, 2 or 3")
    return patch_combinations


def generate_two_conv_combinations(kernel_scales_seq, spatial_dims):
    """
    Generate all possible two layer combinations for a given spatial dimension
    """
    patch_to_partition = create_patch_dict(kernel_scales_seq)
    patch_combinations = generate_patch_combinations(kernel_scales_seq, spatial_dims)
    kernel_scales_seq1 = []
    kernel_scales_seq2 = []
    for patches in patch_combinations:
        temp = []
        temp1 = []
        for p in patches:
            temp.append(patch_to_partition[p][0])
            temp1.append(patch_to_partition[p][1])
        kernel_scales_seq1.append(temp)
        kernel_scales_seq2.append(temp1)
    kernel_scales_seq1 = tuple(set(tuple(tuple(k) for k in kernel_scales_seq1)))
    kernel_scales_seq2 = tuple(set(tuple(tuple(k) for k in kernel_scales_seq2)))

    return kernel_scales_seq1, kernel_scales_seq2


def choose_kernel_size_deterministic(
    x_shape: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    """
    Choose a kernel size deterministically from image size.
    We fix a target number of tokens per axis and choose the kernel size accordingly.
    This target differs between 2D and 3D images
    """
    # This patch dict works with the Well data dimensions.
    # Add functionality to make this adapt to other dimensions if needed later
    patch_dict = {
        0: (1, 1),
        1: (1, 1),
        4: (2, 2),
        8: (4, 2),
        12: (6, 2),
        16: (4, 4),
        24: (6, 4),
        32: (8, 4),
    }
    if len(x_shape) == 1:
        per_axis_tokens = 512 // 16
        H = x_shape[0]
        non_singleton_D = int(H != 1)
        assert H % per_axis_tokens == 0 or H == 1
        h_patch = H // per_axis_tokens
        return (patch_dict[h_patch],)
    elif len(x_shape) == 2:
        per_axis_tokens = 512 // 16
        H, W = x_shape[:2]
        non_singleton_D = int(H != 1) + int(W != 1)
        assert (H % per_axis_tokens == 0 or H == 1) and (
            W % per_axis_tokens == 0 or W == 1
        )
        h_patch = H // per_axis_tokens
        w_patch = W // per_axis_tokens
        return (patch_dict[h_patch], patch_dict[w_patch])
    elif len(x_shape) == 3:
        # per_axis_tokens = 256 // 16
        H, W, D = x_shape[:3]
        non_singleton_D = int(H != 1) + int(W != 1) + int(D != 1)
        if non_singleton_D == 1:
            per_axis_tokens = 512 // 16
        elif non_singleton_D == 2:
            per_axis_tokens = 512 // 16
        else:
            per_axis_tokens = 256 // 16
        assert (
            (H % per_axis_tokens == 0 or H == 1)
            and (W % per_axis_tokens == 0 or W == 1)
            and (D % per_axis_tokens == 0 or D == 1)
        )
        h_patch = H // per_axis_tokens
        w_patch = W // per_axis_tokens
        d_patch = D // per_axis_tokens
        return (patch_dict[h_patch], patch_dict[w_patch], patch_dict[d_patch])
    else:
        raise ValueError("Image size must be 1, 2 or 3 dimensions")


InterpolationType = Literal[
    "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
]


def _cache_pinvs(
    kernel_scales_seq: Tuple[Tuple[int, int], ...],
    interpolation: InterpolationType,
    antialias: bool,
    base_kernel_size: Tuple[int, int],
    spatial_dims: int = 2,
) -> dict:
    """
    Calculate and cache pseudo-inverses of resize matrices for all possible kernels
    """

    pinvs = {}
    for ps in kernel_scales_seq:
        pinvs[ps] = _calculate_pinv(base_kernel_size, ps, interpolation, antialias)
    return pinvs


def _resize(
    x: Tensor,
    shape: Tuple[int, int],
    interpolation: InterpolationType,
    antialias: bool,
    spatial_dims: int = 2,
) -> Tensor:
    """
    Resize tensor x to shape using interpolation
    """

    x_resized = F.interpolate(
        x[None, None, ...],
        shape,
        mode=interpolation,
        antialias=antialias,
    )
    return x_resized[0, 0, ...]


def _calculate_pinv(
    old_shape: Tuple[int, int],
    new_shape: Tuple[int, int],
    interpolation: InterpolationType,
    antialias: bool,
) -> Tensor:
    """
    Calculate pseudo-inverse of resize matrix from old_shape to new_shape
    """

    mat = []
    for i in range(np.prod(old_shape)):
        basis_vec = torch.zeros(old_shape)
        basis_vec[np.unravel_index(i, old_shape)] = 1.0
        mat.append(_resize(basis_vec, new_shape, interpolation, antialias).reshape(-1))
    resize_matrix = torch.stack(mat)
    return torch.linalg.pinv(resize_matrix)


def resize_patch_embed(
    patch_embed: Tensor,
    base_kernel_size: Tuple[int, ...],
    new_patch_size: Tuple[int, ...],
    pinvs: dict,
    spatial_dims: int = 2,
):
    """Resize patch_embed to target resolution via pseudo-inverse resizing"""
    # Return original kernel if no resize is necessary
    if base_kernel_size == new_patch_size:
        return patch_embed

    pinv = pinvs[new_patch_size]
    pinv = pinv.to(patch_embed.device)

    def resample_patch_embed(patch_embed: Tensor):
        resampled_kernel = pinv @ patch_embed.reshape(-1)
        if spatial_dims == 1:
            (h,) = new_patch_size
            return rearrange(resampled_kernel, "(h) -> h", h=h)
        elif spatial_dims == 2:
            h, w = new_patch_size
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)
        else:
            h, w, d = new_patch_size
            return rearrange(resampled_kernel, "(h w d) -> h w d", h=h, w=w, d=d)

    v_resample_patch_embed = vmap(vmap(resample_patch_embed, 0, 0), 1, 1)

    return v_resample_patch_embed(patch_embed)
