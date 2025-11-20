import unittest
from dataclasses import dataclass

import torch

from walrus_steering.models.shared_utils.patch_jitterers import PatchJitterer


@dataclass
class DummyMetadata:
    n_spatial_dims: int


class TestJitterer(unittest.TestCase):
    """Right now these just test that forward jitters at least some of the time and that
    unjitter(jitter(x)) is identity"""

    def setUp(self):
        self.jitterer = PatchJitterer(
            3, (16, 16, 16), num_bcs=3, max_d=3, jitter_patches=True
        )

    def test_inverse_1d_boundary(self):
        metadata = DummyMetadata(1)
        x = torch.randn(1, 1, 3, 16, 1, 1)  # T B C H W D
        mid, jitter_info = self.jitterer(x, torch.tensor([[0, 0]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "1D inverse jitter failed for nonperiodic BC"

    def test_inverse_2d_boundary(self):
        metadata = DummyMetadata(2)
        x = torch.randn(1, 1, 3, 16, 16, 1)
        mid, jitter_info = self.jitterer(x, torch.tensor([[0, 0], [0, 0]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "2D inverse jitter failed for nonperiodic BC"

    def test_inverse_3d_boundary(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = self.jitterer(
            x, torch.tensor([[0, 0], [0, 0], [0, 0]]), metadata
        )
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "3D inverse jitter failed for nonperiodic BC"

    def test_inverse_1d_periodic(self):
        metadata = DummyMetadata(1)
        x = torch.randn(1, 1, 3, 16, 1, 1)
        mid, jitter_info = self.jitterer(x, torch.tensor([[2, 2]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "1D inverse jitter failed for periodic BC"

    def test_inverse_2d_periodic(self):
        metadata = DummyMetadata(2)
        x = torch.randn(1, 1, 3, 16, 16, 1)
        mid, jitter_info = self.jitterer(x, torch.tensor([[2, 2], [2, 2]]), metadata)
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "2D inverse jitter failed for periodic BC"

    def test_inverse_3d_periodic(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = self.jitterer(
            x, torch.tensor([[2, 2], [2, 2], [2, 2]]), metadata
        )
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "3D inverse jitter failed for periodic BC"

    def test_inverse_3d_mixed(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = self.jitterer(
            x, torch.tensor([[0, 0], [2, 2], [0, 0]]), metadata
        )
        y = self.jitterer.unjitter(mid, jitter_info)
        assert torch.allclose(x, y), "3D inverse jitter failed for mixed BC"

    def test_3d_turned_off(self):
        jitterer = PatchJitterer(
            3, (16, 16, 16), num_bcs=3, max_d=3, jitter_patches=False
        )
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        mid, jitter_info = jitterer(x, torch.tensor([[0, 0], [2, 2], [0, 0]]), metadata)
        assert torch.allclose(x, mid), "3D jitter failed for turned off jitterer"

    def test_3d_jittering_nonperiodic(self):
        metadata = DummyMetadata(3)
        x = torch.randn(1, 1, 3, 16, 16, 16)
        counter = 0
        # Jitter can randomly return true... so we'll just check a few times - should be p=(1/patch_size)^d
        for i in range(3):
            mid, jitter_info = self.jitterer(
                x, torch.tensor([[2, 2], [2, 2], [2, 2]]), metadata
            )
            if not torch.allclose(x, mid):
                counter += 1
        assert counter > 0, "3D jitter failed"
