from dataclasses import replace
from functools import reduce
from operator import mul
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from walrus_steering.models.shared_utils.flexi_utils import (
    choose_kernel_size_deterministic,
    choose_kernel_size_random,
)
from walrus_steering.models.shared_utils.mlps import (
    SubsampledLinear,  # Make this use library once lbirary is setup
)
from walrus_steering.models.shared_utils.normalization import RMSGroupNorm
from walrus_steering.models.shared_utils.patch_jitterers import PatchJitterer

from torch.profiler import profile, record_function, ProfilerActivity

def dim_pad(x, max_d):
    """
    Assume T B C are first channels, then see how many spatial dims we need to append/
    """
    squeeze = 0
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    if x.ndim - 3 < max_d:
        x = x.unsqueeze(-1)
        squeeze += 1
    return x, squeeze


class IsotropicModel(nn.Module):
    """
    Naive model that operates at a single dimension with a repeating block.

    Args:
        patch_size (tuple): Size of the input patch
        hidden_dim (int): Dimension of the embedding
        processor_blocks (int): Number of blocks (consisting of spatial mixing - temporal attention)
        n_states (int): Number of input state variables.
    """

    def __init__(
        self,
        encoder,
        decoder,
        processor,
        projection_dim: int = 96,
        intermediate_dim: int = 192,
        hidden_dim: int = 768,
        processor_blocks: int = 8,
        n_states: int = 4,
        drop_path: float = 0.2,
        input_drop: float = 0.1,
        groups: int = 12,
        max_d: int = 3,
        static_axes: bool = False,
        jitter_patches: bool = True,
        weight_tied_axes: bool = True,
        gradient_checkpointing_freq: int = 0,
        causal_in_time: bool = False,
        include_d: List[int] = [2, 3],  # Temporary due to FSDP resume issue
        override_dimensionality: Optional[
            int
        ] = 0,  # Temporary due to FSDP resume issue
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.drop_path = drop_path
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes
        # self.pos_emb = nn.Parameter(torch.randn(16, 1, hidden_dim, 128//16, 128//16, 1)*.02)
        self.dp = np.linspace(0, drop_path, processor_blocks)
        self.space_bag = SubsampledLinear(n_states, projection_dim)
        self.causal_in_time = causal_in_time
        self.static_axes = static_axes
        self.gradient_checkpointing_freq = gradient_checkpointing_freq
        self.override_dimensionality = override_dimensionality
        self.encoder_dummy = nn.Parameter(torch.ones(1)) # for grad checkpointing, see: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/11
        # if (
        #     self.override_dimensionality is not None
        #     and self.override_dimensionality > 0
        # ):
        #     include_d = [self.override_dimensionality]
        self.input_drop = input_drop
        self.dropout_dict = {"1": F.dropout1d,
                             "2": F.dropout2d,
                             "3": F.dropout3d}
        self.patch_jitterer = PatchJitterer(
            stage_dim=projection_dim,
            patch_size=None,
            max_d=self.max_d,
            jitter_patches=jitter_patches,
        )
        self.embed = nn.ModuleDict(
            {
                str(i): encoder(
                    spatial_dims=3,
                    input_dim=projection_dim,
                    inner_dim=intermediate_dim,
                    output_dim=hidden_dim,
                    groups=groups,
                    norm_layer=norm_layer,
                )
                for i in range(1, self.max_d + 1)
                if i in include_d
            }
        )

        self.blocks = nn.ModuleList(
            [
                processor(
                    hidden_dim=hidden_dim,
                    drop_path=self.dp[i],
                    causal_in_time=causal_in_time,
                    gradient_checkpointing=(
                        i % gradient_checkpointing_freq == 0
                        if gradient_checkpointing_freq > 0
                        else False
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(processor_blocks)
            ]
        )
        self.debed = nn.ModuleDict(
            {
                str(i): decoder(
                    input_dim=hidden_dim,
                    inner_dim=intermediate_dim,
                    output_dim=n_states,
                    spatial_dims=3,
                    groups=groups,
                    norm_layer=norm_layer,
                )
                for i in range(1, self.max_d + 1)
                if i in include_d
            }
        )

    def freeze_middle(self):
        # First just turn grad off for everything
        for param in self.parameters():
            param.requires_grad = False
        # Activate for embed/debed layers
        for param in self.readout_head.parameters():
            param.requires_grad = True
        for param in self.space_bag.parameters():
            param.requires_grad = True
        self.debed.out_kernel.requires_grad = True
        self.debed.out_bias.requires_grad = True

    def freeze_processor(self):
        # First just turn grad off for everything
        for param in self.parameters():
            param.requires_grad = False
        # Activate for embed/debed layers
        for param in self.readout_head.parameters():
            param.requires_grad = True
        for param in self.space_bag.parameters():
            param.requires_grad = True
        for param in self.debed.parameters():
            param.requires_grad = True
        for param in self.embed.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def _encoder_forward(
        self, x, state_labels, bcs, metadata, patch_size, dynamic_ks=None, encoder_dummy=None
    ):
        # print("in encoder!", bcs.device, bcs.dtype)
        # n_spatial_dims = metadata.n_spatial_dims
        n_spatial_dims = sum([int(dim!=1) for dim in x.shape[3:]])
        dim_key = str(n_spatial_dims)
        T = x.shape[0]
        # Project into higher dim
        with record_function("space_bag"):
            x = rearrange(x, "t b ... -> (t b) ...")
            x = self.dropout_dict["3"](x, p=self.input_drop/x.shape[1]) # Bonferonni correction for variable fields
            x = rearrange(x, "(t b) ... -> t b ...", t=T)
            x = x * encoder_dummy # NOTE - this deals with a bug in PyTorch's grad checkpointing
            x = rearrange(x, "t b c ... -> t b ... c")
            x = self.space_bag(x, state_labels)
            x = rearrange(x, "t b ... c -> t b c ...")
        # Now encoder
        with record_function("patch_jitter"):
            if (
                hasattr(self.embed[dim_key], "learned_pad")
                and self.embed[dim_key].learned_pad
            ):
                with record_function("patch jitter true"):
                    x, jitter_info = self.patch_jitterer(
                        x,
                        bcs[0],
                        metadata,
                        patch_size=patch_size,
                        learned_pad=self.embed[dim_key].learned_pad,
                        random_kernel=dynamic_ks,
                        base_kernel=self.embed[dim_key].base_kernel_size,
                    )
            else:
                with record_function("patch jitter false"):
                    x, jitter_info = self.patch_jitterer(
                        x, bcs[0], metadata, patch_size=patch_size
                    )

        # Sparse proj
        with record_function("encoder"):
            x, stage_info = self.embed[dim_key](
                x, bcs[0], metadata, random_kernel=dynamic_ks
            )
        return x, stage_info, jitter_info

    def _decoder_forward(self, x, state_labels, stage_info, jitter_info, metadata):
        """Run the decoder and invert the jitter"""
        n_spatial_dims = sum([int(dim!=1) for dim in x.shape[3:]])
        dim_key = str(n_spatial_dims)
        
        x = self.debed[dim_key](x, state_labels, stage_info, metadata)
        if (
            hasattr(self.embed[dim_key], "learned_pad")
            and self.embed[dim_key].learned_pad
        ):
            x = self.patch_jitterer.unjitter(
                x, jitter_info, learned_pad=self.embed[dim_key].learned_pad
            )
        else:
            x = self.patch_jitterer.unjitter(x, jitter_info)
        return x

    def forward(
        self,
        x,
        state_labels,
        bcs,
        metadata,
        proj_axes=None,
        return_att=False,
        train=True,
    ):
        # x - T B C H [W D]
        # state_labels - C
        # bcs - #dims, 2
        # proj axes - #dims - Permutes axes to discourage learning axes - dependent relationships
        # NOTE: HARDCODED IN THIS VERSION SINCE WE'RE USING ONE ENCODER
        # print("just in model!", type(bcs), "state_labels", state_labels.device, state_labels.shape, "x", x.device, x.shape)
        with record_function("intro"):
            if (
                self.override_dimensionality is not None
                and self.override_dimensionality > 0
            ):
                metadata = replace(metadata, n_spatial_dims=self.override_dimensionality)
            n_spatial_dims = metadata.n_spatial_dims
            dim_key = str(n_spatial_dims)
            # Pad to max dims so we can just use 3D convs - same flops, but empirically would be faster
            # to dynamically adjust which conv is used, but more verbose for compiler-friendly version
            x, squeeze_out = dim_pad(x, self.max_d)
            T, B, C = x.shape[:3]
            x_shape = x.shape[3:]

            dynamic_ks = []
            patch_size = []

            # Choose the variable patches if applicable
            if (
                hasattr(self.embed[dim_key], "variable_downsample")
                and (self.embed[dim_key].variable_downsample)
                and self.embed[dim_key].variable_deterministic_ds
            ):
                # support for variable but deterministic downsampling
                dynamic_ks = choose_kernel_size_deterministic(x_shape)
                patch_size = [reduce(mul, k) for k in dynamic_ks]
                # patch_size doesn't matter for the dimension that is higher than the number of spatial dims
                patch_size.extend([0] * (self.max_d - len(patch_size)))

            # support for variable and random downsampling.
            # this will probably not be used in MPPX but a needed feature for dedicated paper
            elif hasattr(self.embed[dim_key], "variable_downsample") and (
                self.embed[dim_key].variable_downsample
            ):
                for _ in range(self.max_d):
                    ks = (
                        choose_kernel_size_random(self.embed[dim_key].kernel_scales_seq)
                        if train
                        else (2, 2)
                    )
                    patch_size.append(ks[0] * ks[1])
                    dynamic_ks.append(ks)
                dynamic_ks = tuple(dynamic_ks)
            # constant downsampling as with hmlp
            else:
                patch_size = [self.embed[dim_key].patch_size] * self.max_d
            # Do not want to overfit to a specific anisotropic setting, so shuffle which axes are used
            if self.static_axes or self.weight_tied_axes:
                axis_order = torch.arange(self.max_d)  #
                if proj_axes is None:
                    axis_order = axis_order[:n_spatial_dims]
                else:
                    axis_order = axis_order[proj_axes]
            else:
                axis_order = torch.randperm(self.max_d)[:n_spatial_dims]

            if dynamic_ks:
                dynamic_ks = tuple([dynamic_ks[axis] for axis in axis_order])
        with record_function("encoder"):
            # Always assume we need to checkpoint the encoder if any checkpointing is on
            if self.gradient_checkpointing_freq > 0:
                x, stage_info, jitter_info = torch.utils.checkpoint.checkpoint(
                    self._encoder_forward,
                    x,
                    state_labels,
                    bcs,
                    metadata,
                    patch_size,
                    dynamic_ks,
                    self.encoder_dummy,
                    use_reentrant=False
                )
            else:
                x, stage_info, jitter_info = self._encoder_forward(
                    x, state_labels, bcs, metadata, patch_size, dynamic_ks, self.encoder_dummy
                )
        with record_function("processor"):
            # Process
            all_att_maps = []
            for blk in self.blocks:
                x, att_maps = blk(x, bcs, axis_order, return_att=return_att)
                all_att_maps += att_maps

        # Decode
        # If not causal, no need to debed all time steps so just take the last one
        with record_function("decoder"):
            if not self.causal_in_time:
                x = x[-1:]

            if self.gradient_checkpointing_freq > 0:
                x = torch.utils.checkpoint.checkpoint(
                    self._decoder_forward,
                    x,
                    state_labels,
                    stage_info,
                    jitter_info,
                    metadata,
                    use_reentrant=False
                )
            else:
                x = self._decoder_forward(
                    x, state_labels, stage_info, jitter_info, metadata
                )

            # De-inflate the extra channels if they were added:
            for _ in range(squeeze_out):
                x = x.squeeze(-1)
        # This is inplace so don't want to mess up next pass
        # if self.override_dimensionality is not None and self.override_dimensionality > 0:
        #     metadata.n_spatial_dims = orig_dim
        # Return T, B, C, H, [W], [D]
        return x  # TODO - Return attention maps for debugging
