import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath

from .swin import create_block_mask, flex_attention


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def build_cswin_op(H, H_sp, W, W_sp, D=1, D_sp=1):
    ## Building this assuming 3D is a possibility
    ## Almost certainly a more efficient approach here, but don't want to spend forever
    def block_mask(b, h, q_idx, kv_idx):
        # Get x, y, z coordinates for q_idx and kv_idx, assuming stride is H, W, D
        z_q, z_kv = q_idx % D, kv_idx % D
        next_q, next_kv = q_idx // D, kv_idx // D
        x_q, x_kv = next_q % W, next_kv % W
        y_q, y_kv = next_q // W, next_kv // W
        # Now make sure each coord is in same window
        z_block_mask = (z_q // D_sp) == (z_kv // D_sp)
        x_block_mask = (x_q // W_sp) == (x_kv // W_sp)
        y_block_mask = (y_q // H_sp) == (y_kv // H_sp)
        return z_block_mask & x_block_mask & y_block_mask

    return block_mask


class LePEAttentionFlex(nn.Module):
    def __init__(
        self,
        dim,
        resolution,
        idx,
        split_size=7,
        dim_out=None,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        qk_scale=None,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.block_mask_func = create_block_mask(
            build_cswin_op(
                self.resolution, self.H_sp, self.resolution, self.W_sp, 1, 1
            ),  # Trailing 1s placeholder for 3D
            B=None,
            H=None,
            Q_LEN=resolution**2,
            KV_LEN=resolution**2,
            # Block size is tricky - need to work through math to get better answer
            # For 2D, we know that we have one direction contiguous and the other at stride
            # "Resolution", so block size needs to be at least < resolution to get any
            # sparsity in that direction. But larger blocks translate to better
            # performance. So we'll start with resolution//4, but this is probably
            # too small.
            #  BLOCK_SIZE=resolution, # Not working on nightly currently - fix later
            _compile=True,
        )  # The 1 is because this is a 2D mask
        # stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    # def im2cswin(self, x):
    #     B, N, C = x.shape
    #     H = W = int(np.sqrt(N))
    #     x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
    #     x = img2windows(x, self.H_sp, self.W_sp)
    #     x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
    #     return x
    # TODO: Where is this used? Because `img2windows` is undefined.

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = int(np.sqrt(N))
        # == W
        # x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = rearrange(
            x, "B (H Hsp W Wsp) C -> (B H W) C Hsp Wsp", Hsp=H_sp, Wsp=W_sp, H=H // H_sp
        )
        lepe = func(x)  ### B', C, H', W'
        lepe = rearrange(
            lepe, "(B H W) C Hsp Wsp -> B (H Hsp W Wsp) C", B=B, H=H // H_sp
        )
        return lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        lepe = self.get_lepe(v, self.get_v)

        # Print shapes of all major tensors
        q, k, v, lepe = map(
            lambda t: rearrange(
                t, "B H (he C) -> B he H C", he=self.num_heads
            ).contiguous(),
            (q, k, v, lepe),
        )
        x = flex_attention(q, k, v, block_mask=self.block_mask_func) + lepe
        x = rearrange(x, "B he H C -> B H (he C)")
        return x


class CSWinBlockFlex(nn.Module):
    def __init__(
        self,
        dim,
        reso,
        num_heads,
        split_size=7,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        last_stage=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList(
                [
                    LePEAttentionFlex(
                        dim,
                        resolution=self.patches_resolution,
                        idx=-1,
                        split_size=split_size,
                        num_heads=num_heads,
                        dim_out=dim,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                    )
                    for i in range(self.branch_num)
                ]
            )
        else:
            self.attns = nn.ModuleList(
                [
                    LePEAttentionFlex(
                        dim // 2,
                        resolution=self.patches_resolution,
                        idx=i,
                        split_size=split_size,
                        num_heads=num_heads // 2,
                        dim_out=dim // 2,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                    )
                    for i in range(self.branch_num)
                ]
            )

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, : C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2 :])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
