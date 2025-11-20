import os
from typing import List

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig
from the_well.data.datamodule import AbstractDataModule
from the_well.data.datasets import BoundaryCondition, WellMetadata
from torch.utils.data import DataLoader, Dataset

from walrus_steering.data.multidatamodule import metadata_aware_collate
from walrus_steering.train import CONFIG_DIR
from walrus_steering.trainer.checkpoints import CheckPointer

#
# Data parameters
#
B = 50  # Dataset size
T = 10  # Number of time steps
N = 32  # Number of spatial points
C = 1  # Number of fields


class LinearModel(nn.Module):
    """Dummy linear model."""

    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(N, N)
        self.causal_in_time = True

    def forward(self, x, *args, **kwargs):
        # x shape: (T, B, C, ...)
        return self.linear(x)


class DummyDataset(Dataset):
    """Dummy dataset."""

    def __init__(self, data):
        assert data.ndim >= 4  # (B, T, N1, ..., Nd, C)
        self.data = data
        self.n_steps_input = data.shape[1] - 1
        spatial_dims = data.shape[2:-1]

        self.metadata = WellMetadata(
            dataset_name="dummy",
            n_spatial_dims=len(spatial_dims),
            grid_type="cartesian",
            spatial_resolution=spatial_dims,
            scalar_names=[],
            constant_scalar_names=[],
            constant_field_names={},
            field_names={i: "" for i in range(data.shape[-1])},
            boundary_condition_types=None,
            n_files=data.shape[0],
            n_trajectories_per_file=[1],
            n_steps_per_trajectory=data.shape[1],
        )

        self.dset_to_metadata = {}
        self.dset_to_metadata["dummy"] = self.metadata
        self.full_trajectory_mode = False

        self.sub_dsets = [self]  # Attribute from MixedWellDataset class

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            "input_fields": self.data[
                idx, : self.n_steps_input
            ],  # Tin x N1 x ... x Nd x C tensor of input trajectory
            "output_fields": self.data[
                idx, self.n_steps_input :
            ],  # Tpred x N1 x ... x Nd x C tensor of output trajectory
            "field_indices": torch.arange(
                self.data.shape[-1]
            ),  # C tensor of field indices
            "metadata": self.metadata,
            "boundary_conditions": BoundaryCondition.PERIODIC.value
            * torch.ones(
                self.metadata.n_spatial_dims, 2
            ),  # Periodic boundary conditions
        }
        return sample


class TestDataModule(AbstractDataModule):
    """Dummy data module."""

    def __init__(self, data):
        super(TestDataModule, self).__init__()

        self.train_dataset = DummyDataset(data[: int(0.8 * B)])
        self.val_dataset = DummyDataset(data[int(0.8 * B) : int(0.9 * B)])
        self.test_dataset = DummyDataset(data[int(0.9 * B) :])
        self.rollout_val_dataset = DummyDataset(data[int(0.8 * B) : int(0.9 * B)])
        self.rollout_test_dataset = DummyDataset(data[int(0.9 * B) :])

        self.batch_size = 10

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=metadata_aware_collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=metadata_aware_collate,
        )

    def rollout_val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.rollout_val_dataset,
            batch_size=self.batch_size,
            collate_fn=metadata_aware_collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=metadata_aware_collate,
        )

    def rollout_test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.rollout_test_dataset,
            batch_size=self.batch_size,
            collate_fn=metadata_aware_collate,
        )

    def val_dataloaders(self) -> List[DataLoader]:
        return [self.val_dataloader()]

    def rollout_val_dataloaders(self) -> List[DataLoader]:
        return [self.rollout_val_dataloader()]

    def test_dataloaders(self) -> List[DataLoader]:
        return [self.test_dataloader()]

    def rollout_test_dataloaders(self) -> List[DataLoader]:
        return [self.rollout_test_dataloader()]


def train(
    trainer_cfg: DictConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    datamodule: AbstractDataModule,
    max_epoch: int,
    **kwargs,
):
    """Train the model for a given number of epochs."""
    trainer = instantiate(
        trainer_cfg.trainer,
        experiment_name="test_trainer",
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        max_epoch=max_epoch,
        device=torch.device("cpu"),
        wandb_logging=False,
        image_validation=False,
        video_validation=False,
        viz_folder="",
        checkpointer=CheckPointer(save_dir=""),
        **kwargs,
    )
    trainer.train()

    _, test_logs = trainer.validation_loop(
        datamodule.test_dataloaders(), valid_or_test="test", full=True
    )
    loss = test_logs["test_dummy/full_RMSE_T=all"]
    return loss


@pytest.mark.skip(reason="to be fixed")
def test_trainer_linear_model():
    """
    Testing the Trainer class on a linear model.
    We vary:
    - data: constant/delta-one dynamics
    - precision: mixed/full
    - prediction_type: delta/full
    Expected behavior is that in all cases the linear model should reach a zero loss easily.
    In practice, we observe that the mixed precision training requires a higher tolerance due to roundoff errors.
    We also note that the normalization of the input/output (spatial mean subtraction) makes in most cases the problem even more trivial.
    """
    prediction_types = ["full", "delta"]
    enable_amps = [True, False]
    datamodules = [
        TestDataModule(torch.ones(B, T, N, C)),  # Constant dynamics
        TestDataModule(
            torch.cumsum(torch.ones(B, T, N, C), dim=1)
        ),  # Delta one dynamics
    ]

    cfg_dir = os.path.abspath(CONFIG_DIR)
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        trainer_cfg = compose(config_name="trainer/debug")
        optimizer_cfg = compose(config_name="optimizer/adam")

    for prediction_type in prediction_types:
        for enable_amp in enable_amps:
            for datamodule in datamodules:
                model = LinearModel()  # Instantiate model
                optimizer = instantiate(
                    optimizer_cfg.optimizer, params=model.parameters()
                )

                loss = train(
                    trainer_cfg=trainer_cfg,
                    model=model,
                    optimizer=optimizer,
                    datamodule=datamodule,
                    max_epoch=100,
                    prediction_type=prediction_type,
                    enable_amp=enable_amp,
                )
                assert (
                    loss < 1e-2 if enable_amp else loss < 1e-5
                )  # Tolerance is higher for mixed precision here


@pytest.mark.skip(reason="to be fixed")
def test_trainer_isotropic_model():
    """
    Testing the Trainer class on a tiny isotropic model.
    We vary:
    - precision: mixed/full
    - data x prediction_type: constant dynamics x full / delta-one dynamics x delta
    Expected behavior is that in all cases the model should reach a zeros loss easily.
    We note that the normalization of the input/output (spatial mean subtraction) makes this all about
    testing that the model can learn to predict nothing.
    """
    settings = [
        ("full", True, TestDataModule(torch.ones(B, T, N, N, C))),
        ("full", False, TestDataModule(torch.ones(B, T, N, N, C))),
        ("delta", True, TestDataModule(torch.cumsum(torch.ones(B, T, N, N, C), dim=1))),
        (
            "delta",
            False,
            TestDataModule(torch.cumsum(torch.ones(B, T, N, N, C), dim=1)),
        ),
    ]
    nb_epochs = 10

    cfg_dir = os.path.abspath(CONFIG_DIR)
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        trainer_cfg = compose(config_name="trainer/debug")
        optimizer_cfg = compose(config_name="optimizer/adam")
        lr_scheduler_cfg = compose(config_name="lr_scheduler/cosine_with_warmup")
        model_cfg = compose(config_name="model/debug")

    for prediction_type, enable_amp, datamodule in settings:
        model = instantiate(
            model_cfg.model,
            processor_blocks=1,
            hidden_dim=24,
            n_states=1,
            groups=2,
            jitter_patches=False,
        )
        optimizer = instantiate(
            optimizer_cfg.optimizer, params=model.parameters(), lr=1e-2
        )
        lr_scheduler = instantiate(
            lr_scheduler_cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=nb_epochs,
            warmup_start_lr=optimizer_cfg.optimizer.lr * 0.1,
            eta_min=optimizer_cfg.optimizer.lr * 0.1,
        )

        loss = train(
            trainer_cfg=trainer_cfg,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            datamodule=datamodule,
            max_epoch=nb_epochs,
            prediction_type=prediction_type,
            enable_amp=enable_amp,
        )
        assert loss < 1e-5
