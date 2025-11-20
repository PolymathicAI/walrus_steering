import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

# from timm.models.registry import register_model
from ..shared_utils.position_biases import RotaryEmbedding, apply_rotary_pos_emb


def flex_attention(*args, **kwargs):
    from torch.nn.attention.flex_attention import flex_attention as flex_attention_impl

    return torch.compile(flex_attention_impl, dynamic=False)(*args, **kwargs)


def create_block_mask(*args, **kwargs):
    from torch.nn.attention.flex_attention import (
        create_block_mask as create_block_mask_impl,
    )

    return create_block_mask_impl(*args, **kwargs)


os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


def window_partition_flex(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, -1, window_size, window_size, C)
    )
    return windows


def window_reverse_flex(windows, window_size, H, W):
    B = windows.shape[0]  # // (H * W // window_size // window_size)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    # return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# def build_swin_mask_and_window_block()
LN_100 = np.log(100)

# def create_swin_score_mod(H, W):
#     def swin_score_mod(score, b, h, q_idx, kv_idx):
#         qx, qy = q_idx % W, q_idx // W
#         kvx, kvy = kv_idx % W, kv_idx // W
#         # scale = torch.clamp(logit_scale[h], max=LN_100).exp()
#         return score + 16 * torch.sigmoid(relative_position_scores[kvy-qy + H//2, kvx-qx+W//2, h])
#     return swin_score_mod


@lru_cache
def create_block_mask_cached(score_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(score_mod, B, H, M, N, device=device)
    return block_mask


def create_swindow_mask(H, W, window_size, shift_size, bcs, device="cuda"):
    # @lru_cache()
    def swindow_mask(b, h, q_idx, kv_idx):
        mask_size = window_size**2

        q_window = q_idx // mask_size
        kv_window = kv_idx // mask_size
        # return q_window == kv_window
        qlocal_idx = q_idx % mask_size
        kvlocal_idx = kv_idx % mask_size

        windows_per_row = W // window_size
        # windows_per_col = H // window_size
        # Get 2D block coordinates
        q_block_x = q_window % windows_per_row
        q_block_y = q_window // windows_per_row
        kv_block_x = kv_window % windows_per_row
        kv_block_y = kv_window // windows_per_row
        # Now get the local coordinates within the block
        qx = q_block_x * window_size + qlocal_idx % window_size
        qy = q_block_y * window_size + qlocal_idx // window_size
        kvx = kv_block_x * window_size + kvlocal_idx % window_size
        kvy = kv_block_y * window_size + kvlocal_idx // window_size

        # print(q_idx, 'block_id:', q_window, ', block x:', q_block_x, ', x:', qx, ', block y:', q_block_y, ', y:', qy)
        # qx, qy = q_idx % W, q_idx // W
        # kvx, kvy = kv_idx % W, kv_idx // W
        x_mask = qx // window_size == kvx // window_size
        y_mask = qy // window_size == kvy // window_size
        # mask = q_idx // mask_size == kv_idx // mask_size
        if shift_size > 0:
            if bcs[0, 0] == 0:
                y_mask &= kvx < H - shift_size
                y_mask &= qx < H - shift_size
            elif bcs[0, 1] == 0:
                x_mask &= kvx < W - shift_size
                y_mask &= qx < H - shift_size
        return x_mask & y_mask

    return swindow_mask


class WindowAttentionFlex(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1, 1))), requires_grad=True
        )
        # swindow_mask = create_swindow_mask(64, 64, 8, 0, torch.tensor([0, 0]))
        # self.swin_mask = create_block_mask(swindow_mask, None, None, 256**2, 256**2, BLOCK_SIZE = 256,
        #                                    device="cuda")
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        # get relative_coords_table
        # relative_coords_h = torch.arange(
        #     -(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32
        # )
        # relative_coords_w = torch.arange(
        #     -(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32
        # )
        # relative_coords_table = (
        #     torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
        #     .permute(1, 2, 0)
        #     .contiguous()
        #     .unsqueeze(0)
        # )  # 1, 2*Wh-1, 2*Ww-1, 2
        # if pretrained_window_size[0] > 0:
        #     relative_coords_table[:, :, :, 0] /= pretrained_window_size[0] - 1
        #     relative_coords_table[:, :, :, 1] /= pretrained_window_size[1] - 1
        # else:
        #     relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        #     relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        # relative_coords_table *= 8  # normalize to -8, 8
        # relative_coords_table = (
        #     torch.sign(relative_coords_table)
        #     * torch.log2(torch.abs(relative_coords_table) + 1.0)
        #     / np.log2(8)
        # )

        # self.register_buffer("relative_coords_table", relative_coords_table)

        # # get pair-wise relative position index for each token inside the window
        # coords_h = torch.arange(self.window_size[0])
        # coords_w = torch.arange(self.window_size[1])
        # coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        # coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative_coords = (
        #     coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # )  # 2, Wh*Ww, Wh*Ww
        # relative_coords = relative_coords.permute(
        #     1, 2, 0
        # ).contiguous()  # Wh*Ww, Wh*Ww, 2
        # relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        # relative_coords[:, :, 1] += self.window_size[1] - 1
        # relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        # self.register_buffer("relative_position_index", relative_position_index)
        self.rope = RotaryEmbedding(dim // (num_heads * len(self.window_size)))
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print("starting flex", x.shape)
        B_, Nw, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, Nw, N, 3, self.num_heads, -1).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        q = torch.clamp(self.logit_scale, LN_100).exp() * q

        freqs = self.rope.get_axial_freqs(*self.window_size)

        q = q.view(B_, self.num_heads, Nw, *self.window_size, -1)
        k = k.view(B_, self.num_heads, Nw, *self.window_size, -1)
        q, k = apply_rotary_pos_emb(freqs, q), apply_rotary_pos_emb(freqs, k)
        q, k = (
            q.view(B_, self.num_heads, Nw * N, -1).contiguous(),
            k.view(B_, self.num_heads, Nw * N, -1).contiguous(),
        )
        v = v.view(B_, self.num_heads, Nw * N, -1).contiguous()

        # attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        # logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        # max_value = torch.log(torch.tensor(1.0 / 0.01, device=self.logit_scale.device))
        # logit_scale = torch.clamp(self.logit_scale, max=max_value).exp()
        # attn = attn * logit_scale

        # relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).squeeze()
        # score_mod = create_swin_score_mod(relative_position_bias_table, self.logit_scale, self.window_size[0], self.window_size[1])

        # .view(
        #    -1, self.num_heads
        # )
        # print('RPE stats', relative_position_bias_table.shape, self.relative_coords_table.shape)
        # relative_position_bias = relative_position_bias_table[
        #     self.relative_position_index.view(-1)
        # ].view(
        #     self.window_size[0] * self.window_size[1],
        #     self.window_size[0] * self.window_size[1],
        #     -1,
        # )  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(
        #     2, 0, 1
        # ).contiguous()  # nH, Wh*Ww, Wh*Ww
        # relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        # print("Final RPE", relative_position_bias.shape, attn.shape)
        # attn = attn + relative_position_bias.unsqueeze(0)

        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
        #         1
        #     ).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        #     attn = self.softmax(attn)

        # attn = self.attn_drop(attn)
        # print shapes of all major tensors
        # print("q", q.shape, "k", k.shape, "v", v.shape)
        x = flex_attention(q, k, v, block_mask=mask).reshape(B_, Nw, N, C)
        # x = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B_, Nw, N, C)

        # x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlockFlex(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        num_heads,
        # input_resolution=224,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
    ):
        super().__init__()
        self.dim = dim
        # self.input_resolution = to_2tuple(input_resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionFlex(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def create_attn_mask(self, H, W, window_size, shift_size, bcs, device):
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1

        if shift_size > 0:
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            if (bcs[0, 0] == 0) and (bcs[0, 1] == 0):
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
            elif (bcs[0, 0] == 1) and (bcs[0, 1] == 0):
                for w in w_slices:
                    img_mask[:, :, w, :] = cnt
                    cnt += 1
            elif (bcs[0, 0] == 0) and (bcs[0, 1] == 1):
                for h in h_slices:
                    img_mask[:, h, :, :] = cnt
                    cnt += 1
            else:
                pass

            mask_windows = window_partition_flex(
                img_mask, window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.reshape(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        return attn_mask

    def forward(self, x, bcs, resolution, mask=None):
        # H, W = self.input_resolution
        H, W = resolution
        B, L, C = x.shape

        assert L == H * W, "input feature has wrong size"
        # device = x.device
        # attn_mask = self.create_attn_mask(
        #     H, W, self.window_size, self.shift_size, bcs, device
        # )
        # swin_mask = create_swindow_mask(H, W, self.window_size, self.shift_size, bcs, device)
        # swin_mask = create_block_mask(swin_mask, None, None, H*W, H*W, BLOCK_SIZE = self.window_size**2,
        #                      device=device)
        shortcut = x
        # x = rearrange(x, 'b (h w) c -> b h w c', h=H)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_flex(
            shifted_x, self.window_size
        )  # B, nW, window_size, window_size, C
        x_windows = x_windows.view(
            B, -1, self.window_size * self.window_size, C
        )  # B, nW, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=mask
        )  # B, nW, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(B, -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_flex(
            attn_windows, self.window_size, H, W
        )  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        # x = rearrange(x, 'b h w c -> b (h w) c')
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x
