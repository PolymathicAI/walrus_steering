import torch
import torch.nn as nn
from einops import rearrange


class hMLP_encoder(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        patch_size=(16, 16),
        output_dim: int = 3,
        input_dim: int = 768,  #
        inner_dim: int = 192,  # Dimension of the internal convs - base is outer/4
        spatial_dims=2,
        groups=12,
    ):
        super().__init__()
        if spatial_dims == 1:
            conv = nn.Conv1d
        elif spatial_dims == 2:
            conv = nn.Conv2d
        elif spatial_dims == 3:
            conv = nn.Conv3d
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.spatial_dims = spatial_dims
        self.spatial_projs = nn.ModuleList(
            [
                torch.nn.Sequential(
                    *[
                        conv(
                            input_dim,
                            inner_dim,
                            kernel_size=4,
                            stride=4,
                            bias=False,
                        ),
                        nn.GroupNorm(groups, inner_dim, affine=True),
                        nn.GELU(),
                    ]
                ),
                torch.nn.Sequential(
                    *[
                        conv(
                            inner_dim,
                            output_dim,
                            kernel_size=4,
                            stride=4,
                            bias=False,
                        ),
                        nn.GroupNorm(groups, output_dim, affine=True),
                        nn.GELU(),
                    ]
                ),
            ]
        )

    def forward(self, x, bcs=None, metadata=None, **kwargs):
        # x is (T, B, C, H, W, D)
        # bcs is (n_dims, 2)
        T = x.shape[0]
        indims = x.ndim
        # Flatten time
        x = rearrange(x, "T B ... -> (T B) ...")  # (T B C H W D) -> (TB C H W D)
        x = x.squeeze((-2, -1))  # (TB C H W D) -> (TB C H [W] [D])
        # Doing this as a loop to eventually permit using stages
        for i, proj in enumerate(self.spatial_projs):
            x = proj(x)
        # Try to add back anything squeezed in the beginning
        x = rearrange(x, "(T B) ... -> T B ...", T=T)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        if x.ndim < indims:
            x = x.unsqueeze(-1)
        return x, {}  # Pad with stage info so we don't need to change later
