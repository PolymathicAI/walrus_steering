from typing import List, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from the_well.data.datasets import BoundaryCondition

from torch.profiler import record_function
class PatchJitterer(nn.Module):
    """Applies random shifts to patches so that error doesn't accumulate in single patches
    For BCs that don't support periodicity, pads the patches with random values before shifting

    Parameters
    ----------
    stage_dim:
        Dimension of the stage
    patch_size:
        Size of the patch
    num_bcs:
        Number of potential boundary conditions #TODO: autopopulate this
    max_d:
        Maximum dimensionality of the data
    jitter_patches:
        Whether to jitter patches or return shaped identity
    """

    def __init__(
        self,
        stage_dim: int,
        patch_size: Optional[Sequence[int]] = None,
        num_bcs: int = 3,
        max_d: int = 3,
        jitter_patches: bool = True,
    ):
        super().__init__()
        self.jitter_patches = jitter_patches
        self.patch_size = patch_size
        self.padding_tokens = nn.Parameter(torch.randn(num_bcs - 1, stage_dim, 1, 1, 1))
        self.max_d = max_d

    def forward(
        self, x, bcs, metadata, patch_size: Optional[Sequence[int]] = None, **kwargs
    ):
        with record_function("PatchJitterer.forward"):
            # x: (T, B, C, H, W, D) - so need to apply to 3D padded data
            # bcs: (n_dims, 2)
            # Allow for identity mapping to simplify code
            if (not self.jitter_patches) and ("learned_pad" not in kwargs):
                return x, {}

            assert (
                (self.patch_size is None) ^ (patch_size is None)
            ), "Must provide EITHER `patch_size` as parameter OR `patch_size` as kwarg to the `forward` call, but not both"
            if patch_size is not None:
                _patch_size: Sequence[int] = patch_size
            else:
                _patch_size = cast(Sequence[int], self.patch_size)

            # This will only work if learned padding is needed even when jitter_patches is False
            T = x.shape[0]
            shape: Sequence[int] = x.shape[3:]
            x = rearrange(x, "t b c h w d -> (t b) c h w d")
            bcs = bcs
            n_dims = metadata.n_spatial_dims
            dim_offset = 3  # Offset by T, B, C
            # If not identity, pad and randomly roll
            paddings: List[int] = []
            # Extra paddings if doing strided convolutions
            extra_paddings: List[int] = [0, 0, 0]
            # If not periodic, apply padding first
            for i in range(self.max_d):
                extra_padding = 0
                if i >= n_dims or shape[i] == 1:
                    axis_padding: List[int] = [0, 0]
                    extra_padding = 0
                else:
                    with record_function("PatchJitterer.forward.first_padding"):
                        if bcs[i][0] == BoundaryCondition["PERIODIC"].value:
                            axis_padding = [0, 0]
                        else:
                            with record_function("PatchJitterer.forward.first_else_pad"):
                                axis_padding = (
                                    [_patch_size[i] // 2, _patch_size[i] // 2]
                                    if self.jitter_patches
                                    else [0, 0]
                                )
                    # print("BC device: ", bcs.device, bcs.dtype)
                    if ("base_kernel" in kwargs) and ("random_kernel" in kwargs):
                        # only true for strided case with learned padding
                        base_kernel1 = kwargs["base_kernel"][i][0]
                        base_kernel2 = kwargs["base_kernel"][i][1]
                        stride1 = kwargs["random_kernel"][i][0]
                        stride2 = kwargs["random_kernel"][i][1]
                        effective_shape = shape[i] + 2 * axis_padding[0]
                        extra_padding = stride1 * stride2 - (
                            (
                                effective_shape
                                - base_kernel1
                                + stride1
                                - base_kernel2 * stride1
                            )
                            % (stride1 * stride2)
                        )
                        extra_padding = extra_padding // 2
                        extra_paddings[i] = extra_padding

                axis_padding_with_extra: List[int] = [
                    axis_padding[0] + extra_padding,
                    axis_padding[1] + extra_padding,
                ]
                paddings = (
                    axis_padding_with_extra + paddings
                )  # Pytorch padding goes [last[start], last[end], ..., first[start], first[end]]

            for i in range(self.max_d):
                indices = 2 * self.max_d - 2 * i - 2, 2 * self.max_d - 2 * i - 1
                axis_pad = [
                    paddings[j] if j in indices else 0 for j in range(len(paddings))
                ]
                if i >= n_dims or sum(axis_pad) == 0:
                    continue
                if bcs[i][0] == BoundaryCondition["PERIODIC"].value:
                    x = F.pad(x, pad=axis_pad, mode="circular")
                else:
                    x = F.pad(x, pad=axis_pad, mode="constant")
            x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
            # Randomly roll each dimension by a random amount < 1 patch
            base_slices = [slice(None)] * len(x.shape)
            roll_quantities, roll_dims = [], []
            for i in range(self.max_d):
                # If we're beyond the number of spatial dims, skip
                if i >= n_dims or shape[i] == 1:
                    continue
                half_patch = (
                    _patch_size[i] // 2 + extra_paddings[i]
                    if self.jitter_patches
                    else extra_paddings[i]
                )
                # Override base slice to specific dimension
                beginning, end = base_slices[:], base_slices[:]
                beginning[i + dim_offset] = slice(None, half_patch)  #
                end[i + dim_offset] = slice(-half_patch, None)
                # apply the padding along the slices (corners are sum of padding tokens)
                with record_function("PatchJitterer.forward.learned bcs"):
                    if bcs[i][0] != BoundaryCondition["PERIODIC"].value:
                        x[tuple(beginning)] += self.padding_tokens[int(bcs[i][0])]
                        x[tuple(end)] += self.padding_tokens[int(bcs[i][1])]
                if self.jitter_patches:
                    # Compute and log the random roll for this dimension
                    from_ = -(half_patch - 1)
                    to_ = half_patch
                    if from_ < to_:
                        roll_rate = int(torch.randint(from_, to_, ()))
                    else:
                        roll_rate = 0
                    # TODO - move this to using random state to avoid compilation issues
                    roll_quantities.append(roll_rate)
                    roll_dims.append(i + dim_offset)
            if self.jitter_patches:
                # Now roll by the randomly sampled values if jitter_patches is true
                x = torch.roll(x, shifts=roll_quantities, dims=roll_dims)
            # Use kwargs for optional compatibility with different versions
            jitter_info = {"paddings": paddings, "rolls": (roll_quantities, roll_dims)}
            return x, jitter_info

    def unjitter(self, x, jitter_info=None, **kwargs):
        if not self.jitter_patches and ("learned_pad" not in kwargs):
            return x
        paddings, rolls = jitter_info["paddings"], jitter_info["rolls"]
        if self.jitter_patches:
            # Reverse the paddings and rolls
            roll_quantities, roll_dims = rolls
            roll_quantities = [-r for r in roll_quantities]
            # Reverse by rolling/padding with negative values
            x = torch.roll(x, shifts=roll_quantities, dims=roll_dims)
        paddings = [-p for p in paddings]
        x = F.pad(x, pad=paddings)
        return x
