# from flash_cosine_sim_attention import flash_cosine_sim_attention
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from ..shared_utils.normalization import RMSGroupNorm
from ..shared_utils.position_biases import (
    ContinuousPositionBias1D,
    RelativePositionBias,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


class AxialTimeAttention(nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-6,
        bias_type="rel",
        gradient_checkpointing=False,
        causal_in_time=False,
        norm_layer=RMSGroupNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSGroupNorm(num_heads, hidden_dim, affine=True)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.input_head = nn.Conv3d(hidden_dim, 3 * hidden_dim, 1)
        self.output_head = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.qnorm = nn.LayerNorm(hidden_dim // num_heads)
        self.knorm = nn.LayerNorm(hidden_dim // num_heads)
        self.bias_type = bias_type
        self.is_causal = causal_in_time
        if bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        elif bias_type == "rotary":
            self.rel_pos_bias = lambda x, y: None
            self.rotary_emb = RotaryEmbedding(hidden_dim // num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(
                n_heads=num_heads, bidirectional=(not causal_in_time)
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def get_rotary_embedding(self, n, device):
        # if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
        #     return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        # self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x, return_att=False):
        # input is t x b x c x h x w
        T, B, C, H, W, D = x.shape
        input = x.clone()
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        x = self.norm1(x)
        x = self.input_head(x)
        x = rearrange(
            x, "(t b) (he c) h w d ->  (b h w d) he t c", t=T, he=self.num_heads
        )
        q, k, v = x.tensor_split(3, dim=-1)
        q, k = self.qnorm(q), self.knorm(k)
        if self.bias_type == "rotary":
            positions = self.get_rotary_embedding(T, self.norm1.weight.device)
            q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))
        # rel_pos_bias returns None if bias_type isn't 'rel'
        rel_pos_bias = self.rel_pos_bias(T, T)
        if return_att:
            if rel_pos_bias is not None:
                att = torch.softmax(
                    (q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1]) + rel_pos_bias,
                    -1,
                )
            else:
                att = torch.softmax(
                    (q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1]), -1
                )
        if rel_pos_bias is not None:
            # Can't use
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=rel_pos_bias)
        else:
            x = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(), is_causal=True
            )
        x = rearrange(x, "(b h w d) he t c -> (t b) (he c) h w d", h=H, w=W, d=D)
        x = self.output_head(x)
        x = rearrange(x, "(t b) c h w d-> t b c h w d", t=T)
        output = self.drop_path(x * self.gamma[None, None, :, None, None, None]) + input
        if return_att:
            return output, [att, rel_pos_bias]
        return output, []
