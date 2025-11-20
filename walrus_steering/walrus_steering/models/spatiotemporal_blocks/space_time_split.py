from functools import partial
from typing import Callable

import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from walrus_steering.models.shared_utils.normalization import RMSGroupNorm


class SpaceTimeSplitBlock(nn.Module):
    """
    Operates similar to standard MHSA -> Inverted Bottleneck but with ConvNext
    block replacing linear part.

    Note: HYDRA is instantiating space_block and time_block as functools.partial functions
    so parameters that aren't shared are pre-set based on the config files.
    """

    def __init__(
        self,
        space_mixing,
        time_mixing,
        channel_mixing,
        hidden_dim=768,
        drop_path=0.0,
        gradient_checkpointing=False,
        causal_in_time=False,
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.space_mixing = space_mixing(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
            norm_layer=norm_layer,
        )
        self.time_mixing = time_mixing(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            gradient_checkpointing=gradient_checkpointing,
            causal_in_time=causal_in_time,
            norm_layer=norm_layer,
        )
        self.channel_mixing = channel_mixing(hidden_dim=hidden_dim)
        self.causal_in_time = causal_in_time

    def forward(self, x, bcs, axis_order, return_att=False):
        # input is t x b x c x h x w
        T, B, C, H, W, D = x.shape
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly
            wrapped_temporal = partial(self.time_mixing, return_att=return_att)
            x, t_att = checkpoint(wrapped_temporal, x, use_reentrant=False)
        else:
            x, t_att = self.time_mixing(x, return_att=return_att)  # Residual in block
        # Temporal handles the rearrange so still is t x b x c x h x w
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        if self.gradient_checkpointing:
            # kwargs seem to need to be passed explicitly
            wrapped_spatial = partial(self.space_mixing, return_att=return_att)
            x, s_att = checkpoint(
                wrapped_spatial, x, bcs, axis_order, use_reentrant=False
            )
        else:
            x, s_att = self.space_mixing(
                x, bcs, axis_order, return_att=return_att
            )  # Convnext has the residual in the block
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        # MLP input is channels last - #TODO redefine as 1x1 conv to avoid reshape
        x = self.channel_mixing(
            x
        )  # Currently set to identity, but needs to be reshaped generally

        if return_att:
            return x, t_att + s_att  # t_att, t_bias, x_att, x_bias, y_att, y_bias
        else:
            return x, []
