import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath

from ..shared_utils.mlps import MLP

# Replace with model path later
from ..shared_utils.position_biases import (
    ContinuousPositionBias1D,
    RelativePositionBias,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


class AxialAttentionBlock(nn.Module):
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
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_d = max_d
        self.weight_tied_axes = weight_tied_axes
        self.norm1 = nn.GroupNorm(num_heads, hidden_dim, affine=True)
        self.norm2 = nn.GroupNorm(num_heads, hidden_dim, affine=True)
        self.gamma_att = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.gamma_mlp = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((hidden_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )

        self.input_heads = nn.ModuleList(
            [nn.Conv3d(hidden_dim, 3 * hidden_dim, 1) for _ in range(max_d)]
        )
        self.output_heads = nn.ModuleList(
            [nn.Conv3d(hidden_dim, hidden_dim, 1) for _ in range(max_d)]
        )
        # self.output_head = nn.Conv3d(hidden_dim, hidden_dim, 1)
        self.qnorms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim // num_heads) for _ in range(max_d)]
        )
        self.knorms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim // num_heads) for _ in range(max_d)]
        )
        if False and bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif False and bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        elif True or bias_type == "rotary":
            # self.pos_emb = None
            self.rotary_emb = RotaryEmbedding(hidden_dim // num_heads)
            # self.register_buffer("pos_emb", None, persistent=False)
        else:
            self.rel_pos_biases = nn.ModuleList(
                [RelativePositionBias(n_heads=num_heads) for _ in range(3)]
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = MLP(hidden_dim)
        self.mlp_norm = nn.GroupNorm(num_heads, hidden_dim, affine=True)
        # self.pre_at_ln
        # self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((hidden_dim)),
        #                     requires_grad=True) if layer_scale_init_value > 0 else None

    def get_rotary_embedding(self, n, device):
        # if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
        #     return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        # self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def spatial_forward(self, x, bcs, axis_index, model_index, return_att=False):
        # Get shapes to use for relative position bias
        B, C, H, W, D = x.shape
        shapes = [H, W, D]
        # If the weights are tied across axes, then we don't need to shuffle the axes.
        if self.weight_tied_axes:
            model_index = 0
        all_inds = ["h", "w", "d"]
        remainder_inds = list(filter(lambda x: x != all_inds[axis_index], all_inds))
        forward_string = f'b he h w d c -> (b {" ".join(remainder_inds)}) he {all_inds[axis_index]} c'
        backward_string = f'(b {" ".join(remainder_inds)}) he {all_inds[axis_index]} c -> b (he c) h w d'
        # Apply the input head and split into Q, K, V
        x = self.input_heads[model_index](x)
        x = rearrange(x, "b (he c) h w d ->  b he h w d c", he=self.num_heads)
        q, k, v = x.tensor_split(3, dim=-1)
        # Apply QK norm and split by heads
        q, k = self.qnorms[model_index](q), self.knorms[model_index](k)
        qx, kx, vx = map(lambda x: rearrange(x, forward_string), [q, k, v])
        # Rel pos bias logic - currently rotary which doesn't know about BCs
        positions = self.get_rotary_embedding(
            shapes[axis_index], self.norm1.weight.device
        )
        qx, kx = map(lambda t: apply_rotary_pos_emb(positions, t), (qx, kx))
        # Complicated return mask logic
        xx = F.scaled_dot_product_attention(
            qx.contiguous(), kx.contiguous(), vx.contiguous()
        )
        # print('pre out rearrange', axis_index, xx.shape)
        if axis_index == 0:
            xx = rearrange(xx, backward_string, w=W, d=D)
        elif axis_index == 1:
            xx = rearrange(xx, backward_string, h=H, d=D)
        else:
            xx = rearrange(xx, backward_string, h=H, w=W)
        xx = self.output_heads[model_index](xx)
        # print('spatial forward size!', axis_index, xx.shape)
        return xx

    def forward(self, x, bcs, axis_order, return_att=False):
        # input is t x b x c x h x w
        B, C, H, W, D = x.shape
        if W == 1:
            ndims = 1
        elif D == 1:
            ndims = 2
        else:
            ndims = 3
        input = x.clone()
        x = self.norm1(x)
        out = torch.zeros_like(x)
        for axis_index, model_index in enumerate(axis_order):
            out = out + self.spatial_forward(
                x, bcs, axis_index, model_index, return_att=return_att
            )

        x = out / ndims  # Normalize to number of dimensions - IDK if necessary
        x = self.drop_path(x * self.gamma_att[None, :, None, None, None]) + input

        # MLP - factor this out since the logic is mostly the same anyway
        input = x.clone()
        x = self.mlp_norm(x)
        x = rearrange(x, "b c h w d -> b h w d c")
        x = self.mlp(x)
        x = rearrange(x, "b h w d c -> b c h w d")

        output = input + self.drop_path(self.gamma_mlp[None, :, None, None, None] * x)
        if return_att:
            return output, []
        else:
            return output, []
