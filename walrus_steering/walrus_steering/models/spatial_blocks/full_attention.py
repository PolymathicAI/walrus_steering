from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

# Replace with model path later
from ..shared_utils.lr_rope_temporary import RotaryEmbedding, apply_rotary_emb
from ..shared_utils.normalization import RMSGroupNorm
from ..shared_utils.position_biases import (
    ContinuousPositionBias1D,
    RelativePositionBias,
)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class FullAttention(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        max_d=3,
        weight_tied_axes=True,
        gradient_checkpointing=False,
        norm_layer: Callable = RMSGroupNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes
        self.norm1 = RMSGroupNorm(num_heads, hidden_dim, affine=True)
        self.gamma_att = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.fused_dims = (
            4 * hidden_dim,
            hidden_dim,
            hidden_dim,
            hidden_dim,
        )  # FF, Q, K, V
        self.fused_ff_qkv = nn.Linear(hidden_dim, sum(self.fused_dims))

        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(2 * hidden_dim, hidden_dim))
        self.attn_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
        self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        if False and bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif False and bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        elif True or bias_type == "rotary":
            # self.pos_emb = None
            self.rotary_emb = RotaryEmbedding(
                hidden_dim // num_heads // 4, freqs_for="pixel", max_freq=256
            )  # Do divide by dimension
            # self.register_buffer("pos_emb", None, persistent=False)
        else:
            self.rel_pos_biases = nn.ModuleList(
                [RelativePositionBias(n_heads=num_heads) for _ in range(3)]
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def get_rotary_embedding(self, n, device):
        # if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
        #     return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        # self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, bcs, axis_order, return_att=False):
        # input is t x b x c x h x w
        B, C, H, W, D = x.shape

        input = x.clone()
        x = self.norm1(x)

        fused_ff_qkv = rearrange(x, "b c h w d -> b h w d c")
        ff, q, k, v = self.fused_ff_qkv(fused_ff_qkv).split(self.fused_dims, dim=-1)

        # Split into heads and process q, k
        q, k, v = map(
            lambda t: rearrange(t, "b h w d (he c) -> b he h w d c", he=self.num_heads),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        pos_emb = self.rotary_emb.get_axial_freqs(H, W, D)
        q, k = map(lambda t: apply_rotary_emb(pos_emb, t), (q, k))
        q, k, v = map(
            lambda t: rearrange(t, "b he h w d c -> b he (h w d) c"), (q, k, v)
        )
        att = F.scaled_dot_product_attention(q, k, v)
        att = rearrange(att, "b he (h w d) c -> b h w d (he c)", h=H, w=W)
        att_out = self.attn_out(att)
        x = self.drop_path(self.gamma_att * (att_out + self.ff_out(ff)))
        x = rearrange(x, "b h w d c -> b c h w d") + input

        return x, []
