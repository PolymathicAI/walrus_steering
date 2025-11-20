import torch


def test_import():
    import temporary_mppx_name  # noqa: F401
    from walrus_steering.models.shared_utils.mlps import MLP

    model = MLP(3)
    model(torch.randn(1, 3))
