import gc
import glob
import logging
import multiprocessing
import os
import pickle as pkl
import shutil
import time
import warnings
from collections import Counter
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from subprocess import CalledProcessError
from typing import Any, Callable, Literal, Optional

import numpy as np # type: ignore
import torch # type: ignore
import torch.distributed as dist # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP # type: ignore
import wandb # type: ignore
from the_well.benchmark.metrics import ( # type: ignore
    long_time_metrics,
    make_video,
    plot_all_time_metrics,
    validation_metric_suite,
    validation_plots,
)
from the_well.data.datamodule import AbstractDataModule # type: ignore
from the_well.data.datasets import WellDataset, flatten_field_names # type: ignore
from torch.amp.grad_scaler import GradScaler # type: ignore
from torch.distributed.fsdp.api import ShardingStrategy # type: ignore
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler # type: ignore
from torch.utils.data import DataLoader # type: ignore

from walrus_steering.data.well_to_multi_transformer import AbstractFormatter
from walrus_steering.trainer.checkpoints import CheckPointLoader

logger = logging.getLogger(__name__)

from torch.profiler import profile, record_function, ProfilerActivity # type: ignore

warnings.filterwarnings('ignore', message='RuntimeWarning: Only axis-aligned boundary fully supported*')

def trace_handler(p):
    # output = p.key_averages().table(sort_by=sort_by_keyword, row_limit=10)
    # print(output)
    p.export_chrome_trace(f"trace_{p.step_num}_{torch.distributed.get_rank()}.json")

def expand_mask_to_match(mask, target):
    """Expand mask of shape B H [W D] 1 to
    broadcast with tensor of given shape B T H [W D] C"""
    T = target.shape[1]
    C = target.shape[-1]
    expansion_tuple = (
        -1,
        T,
    )
    expansion_tuple = expansion_tuple + (-1,) * (len(target.shape) - 3) + (C,)
    mask = mask.unsqueeze(1).expand(*expansion_tuple)
    return mask


def get_grad_norm_local(model) -> torch.Tensor:
    """Computes grad norm for the specific device

    From https://github.com/pytorch/pytorch/issues/88621"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


def get_grad_norm_fsdp(
    model, rank, world_size, sharding_strategy=ShardingStrategy.FULL_SHARD
) -> torch.Tensor:
    """Combines grad norm for the specific device

    From https://github.com/pytorch/pytorch/issues/88621"""
    local_norm = get_grad_norm_local(model)
    op = torch.distributed.ReduceOp.SUM
    return_norm = local_norm.clone().detach().requires_grad_(False) ** 2
    dist.all_reduce(return_norm, op=op)
    if sharding_strategy == ShardingStrategy.NO_SHARD:
        return_norm = return_norm / world_size
    return return_norm**0.5


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5
    
def pad_validation_data(dataloaders, max_length=None):
    """ FSDP is synchronous and collect_all_params is throwing unexpected
    errors, so we are instead padding such that each GPU can go through same amount
    of data even if unnecessary in a normal setting"""
    world_size = dist.get_world_size()
    dataloaders = sorted(dataloaders, key=lambda x: -len(x))
    loader_lengths = [len(dl) if not max_length else min(len(dl), max_length)
                        for dl in dataloaders]
    counts = Counter(loader_lengths)
    ind = 0
    new_dataloaders = []
    for val, count in counts.items():
        rounded_count = count // world_size * world_size
        padding_needed = count - rounded_count
        sub_dl = dataloaders[ind:ind+count]
        new_dataloaders += sub_dl + [sub_dl[-1]] * padding_needed
        ind += count
    return new_dataloaders




    rounded_num_ds = len(dataloaders) // world_size * world_size



@dataclass
class SamplewiseNormalizationStats:
    """
    Class to store normalization statistics for a samplewise normalization.
    """

    sample_mean: torch.Tensor
    sample_std: torch.Tensor
    # sample_norm: torch.Tensor
    delta_mean: torch.Tensor
    delta_std: torch.Tensor
    # delta_norm: torch.Tensor
    epsilon: float = 1e-5


class SamplewiseRevNormalization:
    """
    Module computes normalization and inverts normalization for computing loss statistics

    Data assumed to be in T B C H [W D] format for consistency with MPP repo models
    """

    def compute_stats(
        self, x: torch.Tensor, metadata, epsilon: float = 5e-6
    ) -> SamplewiseNormalizationStats:
        """
        Compute normalization statistics for a batch of data.

        Note - channels are always assumed to be in the 2 dimension.
        """
        # x - T B C H [W D] - MPP format
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            dims = self.get_dims_from_metadata(metadata)
            # Compute samplewise mean and std
            sample_std, sample_mean = x.square().mean(dims, keepdim=True).sqrt(), 0
            sample_std = torch.maximum(sample_std, torch.tensor(epsilon, device=sample_std.device))
            # Compute delta mean and std
            assert x.shape[0] > 1, "Cannot compute delta with only one time frame"
            deltas = x[1:] - x[:-1]  # u_t - u_{t-1}
            delta_std, delta_mean = deltas.square().mean(dims, keepdim=True).sqrt(), 0
            delta_std = torch.maximum(delta_std, torch.tensor(epsilon, device=delta_std.device))
            return SamplewiseNormalizationStats(
                sample_mean, sample_std, delta_mean, delta_std, epsilon
            )

    def get_dims_from_metadata(self, metadata) -> tuple:
        # data assumed to be in T B C H [W D] - MPP format
        if metadata.n_spatial_dims == 1:
            return (0, 3)
        elif metadata.n_spatial_dims == 2:
            return (0, 3, 4)
        else:
            assert metadata.n_spatial_dims == 3
            return (0, 3, 4, 5)

    def normalize_stdmean(
        self,
        x: torch.Tensor,
        stats: SamplewiseNormalizationStats,
        reshape_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Normalize data using the samplewise mean and std.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            n_channels = x.shape[2]
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return x / stats.sample_std[:, :, :n_channels]

    def normalize_delta(
        self,
        x: torch.Tensor,
        stats: SamplewiseNormalizationStats,
        reshape_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Normalize data using the delta mean and std.
        """
        # x - T B C H [W D]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n_channels = x.shape[2]
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return x / stats.delta_std[:, :, :n_channels]

    def denormalize_stdmean(
        self, x: torch.Tensor, stats: SamplewiseNormalizationStats
    ) -> torch.Tensor:
        """
        Denormalize data using the samplewise mean and std.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n_channels = x.shape[2]
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return x * stats.sample_std[:, :, :n_channels]

    def denormalize_delta(
        self, x: torch.Tensor, stats: SamplewiseNormalizationStats
    ) -> torch.Tensor:
        """
        Denormalize data using the delta mean and std.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n_channels = x.shape[2]
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return x * stats.delta_std[:, :, :n_channels]


def normalize_target(
    y_ref: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    formatter: AbstractFormatter,
    metadata: Any,
    device: Any,
) -> torch.Tensor:
    """Helper function to assist in computing targets since this is done in multiple paths.

    1. Transform means/stds from the model format to the validation format
    2. Moves target to device
    3. Normalizes target using reformatted mean/std
    """
    with torch.autocast(device_type=device.type, enabled=False):
        y_ref = y_ref.float()
        # mean = mean.float()
        std = std.float()
        std = formatter.process_output(std, metadata)[..., : y_ref.shape[-1]]
        y_ref = y_ref / std
        return y_ref


class Trainer:
    grad_scaler: GradScaler

    def __init__(
        self,
        experiment_name: str,
        viz_folder: str,
        experiment_folder: str,
        formatter: Callable,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        prediction_type: str,
        # validation_suite: list,
        max_epoch: int,
        val_frequency: int,
        rollout_val_frequency: int,
        max_rollout_steps: int,
        short_validation_length: int,
        checkpointer: CheckPointLoader,
        num_time_intervals: int,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        device_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
        sampling_rank_strategy: Literal["gpu", "node", "world"] = "node",
        distribution_type: str = "local",
        rank: int = 0,
        world_size: int = 1,
        enable_amp: bool = False,
        amp_type: str = "float16",  # bfloat not supported in FFT
        grad_acc_steps: int = 1,
        clip_gradient: float = 0.0,
        loss_multiplier: float = 1.0,
        minimum_context: int = 1,
        masked_loss_for_objects: bool = True,
        video_validation: bool = False,
        image_validation: bool = False,
        save_raw_predictions: bool = False,
        gradient_log_level: int = 0,
        log_interval: int = 10,
        wandb_logging: bool = True,
        start_epoch: int = 1,
        lr_scheduler_per_step: bool = False,
        start_val_loss: Optional[float] = None,
    ):
        """
        Class in charge of the training loop. It performs train, validation and test.

        Parameters
        ----------
        experiment_name:
            The name of the training experiment to be run
        viz_folder:
            The folder where visualizations are saved
        formatter:
            Callable that initializes formatter object that maps between Well and model formats.
        model:
            PyTorch model used for training.
        datamodule:
            A datamodule that provides dataloaders for each split (train, valid, and test)
        optimizer:
            A Pytorch optimizer to perform the backprop (e.g. Adam)
        loss_fn:
            A loss function that evaluates the model predictions to be used for training
        prediction_type:
            The type of prediction to make. Options are "delta" or "full". "delta" predicts the change in the
            field from the previous timestep. "full" predicts the full field at the next timestep.
            This only affects training since validation losses are computed on reconstructed fields
            either way.
        max_epoch:
            Number of epochs to train the model.
            One epoch correspond to a full loop over the datamodule's training dataloader
        val_frequency:
            The frequency in terms of number of epochs to perform the validation
        rollout_val_frequency:
            The frequency in terms of number of epochs to perform the rollout validation
        max_rollout_steps:
            The maximum number of timesteps to rollout the model during long validation.
        num_time_intervals:
            The number of time intervals to bin the loss over for logging purposes.
        lr_scheduler:
            A Pytorch learning rate scheduler to update the learning rate during training
        device:
            A Pytorch device (e.g. "cuda" or "cpu")
        device_mesh:
            Device mesh used for distributed training
        distribution_type:
            The type of distribution to use. Options are "local", "ddp", "fsdp", "hsdp"
        rank:
            The rank of the current GPU in the PyTorch world.
        world_size:
            The total number of GPUs in the PyTorch world
        enable_amp:
            A boolean flag to enable automatic mixed precision training
        amp_type:
            The type of automatic mixed precision to use. Options are "float16" or "bfloat16"
        grad_acc_steps:
            The number of gradient accumulation steps to perform between optimizer steps
        clip_gradient:
            The maximum gradient norm to clip to. If 0, no clipping is performed.
        loss_multiplier:
            A float to multiply the loss by before backpropagating. Useful for scaling loss for different
            datasets or loss functions.
        video_validation:
            A boolean flag to enable saving rollouts to disk during validation
        image_validation:
            A boolean flag to enable saving images to disk during validation
        save_raw_predictions:
            A boolean flag to enable saving raw predictions to disk during validation
        gradient_log_level:
            An integer representing the level of gradient logging. 0 is no logging, 1 full synced gradient only
        log_interval:
            An integer representing how often to log training information. This results in gpu-cpu sync.
        wandb_logging:
            A boolean flag to enable logging to Weights and Biases        checkpoint_frequency:
            An integer representing after how many epochs training checkpoint is saved
        start_epoch:
            The epoch to start training from. Useful for resuming training.
        lr_scheduler_per_step:
            A boolean flag to update the learning rate after each optimizer step instead of each epoch.
        start_val_loss:
            The validation loss to start from. Useful for resuming training.
        """
        self.experiment_name = experiment_name
        self.viz_folder = viz_folder
        self.experiment_folder = experiment_folder
        self.wandb_logging = wandb_logging
        self.video_validation = video_validation
        self.image_validation = image_validation
        self.gradient_log_level = gradient_log_level
        self.log_interval = log_interval
        self.device = device
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        self.validation_suite = validation_metric_suite + [self.loss_fn]
        assert (
            max_epoch + 1 > start_epoch
        ), f"Expect to train for at least one epoch but request starting from {start_epoch} until {max_epoch} epochs."
        # These starting parameters are just for resuming runs
        self.start_epoch = start_epoch
        self.start_val_loss = start_val_loss
        # Run logistics
        self.max_epoch = max_epoch
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps
        self.short_validation_length = short_validation_length
        self.num_time_intervals = num_time_intervals
        self.enable_amp = enable_amp
        self.grad_acc_steps = grad_acc_steps
        self.clip_gradient = clip_gradient
        self.loss_multiplier = loss_multiplier
        self.minimum_context = minimum_context
        self.lr_scheduler_per_step = lr_scheduler_per_step
        self.masked_loss_for_objects = masked_loss_for_objects
        self.save_raw_predictions = save_raw_predictions
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.checkpointer = checkpointer
        # If local or DDP, can use standard grad scaler
        if distribution_type.upper() in ["LOCAL", "DDP"]:
            self.grad_scaler = torch.GradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )
        # Otherwise need sharded version.
        else:
            self.grad_scaler = ShardedGradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )

        self.device_mesh = device_mesh
        self.is_distributed = device_mesh is not None
        self.distribution_type = distribution_type
        self.rank = rank
        self.world_size = world_size
        self.sampling_rank_strategy = sampling_rank_strategy
        # Get derived rank info about which nodes must be synced from the device mesh
        if self.device_mesh is not None and "fsdp" in self.device_mesh.mesh_dim_names:
            self.sync_group = self.device_mesh.get_group(mesh_dim="fsdp")
            self.sync_group_size = self.sync_group.size()
        else: # Local or DDP
            self.sync_group = None
            self.sync_group_size = 1
        # Local or DDP do not need all-gather to do forward pass
        self.sync_group_rank = self.rank // self.sync_group_size
        self.rank_in_sync_group = self.rank % self.sync_group_size
        self.num_sync_groups = self.world_size // self.sync_group_size
        if self.sampling_rank_strategy == "gpu":
            self.sampling_rank = self.rank
        elif self.sampling_rank_strategy == "node":
            self.sampling_rank = self.sync_group_rank
        else:
            self.sampling_rank = 0

        self.dset_metadata = self.datamodule.train_dataset.dset_to_metadata
        self.revin = SamplewiseRevNormalization()

        self.formatter_dict = {}
        # Initial formatter for each dataset - right now these are all identical
        # but we might want to differentiate them in the future.
        for dset_name, metadata in self.dset_metadata.items():
            self.formatter_dict[metadata.dataset_name] = formatter()

    def save_model_if_necessary(
        self, epoch: int, validation_loss: float, last: bool = False
    ) -> Optional[Future]:
        """Save the model checkpoint.
        Force checkpointing if last.
        """
        checkpoint_future = self.checkpointer.save_if_necessary(
            self.model, self.optimizer, validation_loss, epoch, force=last
        )
        return checkpoint_future

    def rollout_model(self, model, batch, formatter, train=True, fake_pass=False):
        """Rollout the model for as many steps as we have data for.

        predict_normalized: bool - If true, output normalized prediction. During one-step training,
            predict normalized values to reduce precision issues/extra FLOPs. During rollout,
            denormalize the output for loss calculation. If multiple steps used during training,
            throw error because not currently supported.
        """

        metadata = batch["metadata"]
        batch = {
            k: v.to(self.device, non_blocking=True)
            if k not in {"metadata", "boundary_conditions"}
            else v
            for k, v in batch.items()
        }

        # Extract mask and move to device for loss eval
        if (
            self.masked_loss_for_objects
            and "mask" in batch["metadata"].constant_field_names[0]
        ):
            mask_index = batch["metadata"].constant_field_names[0].index("mask")
            mask = batch["constant_fields"][..., mask_index : mask_index + 1]
            mask = mask.to(self.device, dtype=torch.bool, non_blocking=True)
        else:
            mask = None

        inputs, y_ref = formatter.process_input(
            batch,
            causal_in_time=model.causal_in_time,
            predict_delta=self.prediction_type == "delta",
            train=train,
        )
        # Inputs T B C H [W D], y_ref B T H [W D] C
        # If causal, during training don't include initial context in rollout length
        T_in = batch["input_fields"].shape[1]
        if model.causal_in_time:
            max_rollout_steps = self.max_rollout_steps + (T_in - 1)
        else:
            max_rollout_steps = self.max_rollout_steps
        rollout_steps = min(
            y_ref.shape[1], max_rollout_steps
        )  # Number of timesteps in target
        train_rollout_limit = T_in if (train and model.causal_in_time) else 1
        if rollout_steps > train_rollout_limit and train:
            raise ValueError("Multiple step prediction in train mode not yet supported")
        y_ref = y_ref[:, :rollout_steps]
        # Create a moving batch of one step at a time
        moving_batch = batch
        y_preds = []
        if fake_pass:
            return y_ref, y_ref
        # Rollout the model - Causal in time gets more predictions from the first step
        for i in range(train_rollout_limit - 1, rollout_steps):
            # Don't fill causal_in_time here since that only affects y_ref
            inputs, _ = formatter.process_input(moving_batch)
            inputs = list(inputs)
            with torch.no_grad():
                normalization_stats = self.revin.compute_stats(inputs[0], metadata)
            # NOTE - Currently assuming only [0] (fields) needs normalization
            normalized_inputs = inputs[:]  # Map type bugs out
            normalized_inputs[0] = self.revin.normalize_stdmean(
                normalized_inputs[0], normalization_stats
            )
            y_pred = model(normalized_inputs[0], normalized_inputs[1], normalized_inputs[2].tolist(), metadata=metadata)
            # During validation, don't maintain full inner predictions
            if not train and model.causal_in_time:
                y_pred = y_pred[-1:]  # y_pred is T first, y_ref is not
            # Train used normalized values to avoid precision loss
            # Validation on the other hand, reconstructs predictions on original scale
            if train:
                pass  # Do nothing since we're computing loss on predicted value and normalizing "ref"
            elif self.prediction_type == "delta":
                # y_pred - (T_all or T=-1 depending on causal or not), B, C, H, [W, D]. Different from y_ref
                with torch.autocast(
                    self.device.type, enabled=False, dtype=self.amp_type
                ):
                    y_pred = inputs[0][
                        -y_pred.shape[0] :
                    ].float() + self.revin.denormalize_delta(
                        y_pred, normalization_stats
                    )  # Unnormalize delta and add to input
            elif self.prediction_type == "full":
                y_pred = self.revin.denormalize_stdmean(y_pred, normalization_stats)
            else:
                raise ValueError(
                    f"Invalid prediction type {self.prediction_type}. Valid types are delta/full"
                )
            y_pred = formatter.process_output(y_pred, metadata)[
                ..., : y_ref.shape[-1]
            ]  # Cut off constant channels

            # TODO - redo losses to accept losses since this will be more efficient there
            if mask is not None:
                mask_pred = expand_mask_to_match(mask, y_pred)
                y_pred.masked_fill_(mask_pred, 0)
            # The person reading this might ask - why is the mask in-place above and a copy below? 
            # The answer is that job failed using in_place below but not above. 
            y_pred = y_pred.masked_fill(~batch["field_mask"], 0.)
            # y_pred.masked_fill_(batch["field_mask"], 0.)  # Mask out unused fields
            # If not last step, update moving batch for autoregressive prediction
            # TODO - for anyone updating this later, it's the primary reason why
            # multiple steps isn't currently supported since we want to recompute
            # normalization stats at each step, but also want to compute loss
            # on normalized values
            if i != rollout_steps - 1:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
                )
            # For causal models, we get use full predictions for the first batch and
            # incremental predictions for subsequent batches - concat 1:T to y_ref for loss eval
            # TODO - test this works - currently getting non-causal working then looping back
            if model.causal_in_time and i == train_rollout_limit - 1:
                y_preds.append(y_pred)
            else:
                y_preds.append(y_pred[:, -1:])
        y_pred_out = torch.cat(y_preds, dim=1)
        # Post-processing y_ref depending on train - if train, normalize y_ref before loss calc
        # If not train, we already denormalized the prediction
        if train:
            mean = (
                normalization_stats.sample_mean
                if self.prediction_type == "full"
                else normalization_stats.delta_mean
            )
            std = (
                normalization_stats.sample_std
                if self.prediction_type == "full"
                else normalization_stats.delta_std
            )
            y_ref = normalize_target(y_ref, mean, std, formatter, metadata, self.device)
        if mask is not None:
            mask_ref = expand_mask_to_match(mask, y_ref)
            y_ref.masked_fill_(mask_ref, 0)

        del moving_batch, batch, mask  # Free up batch memory when done
        return y_pred_out, y_ref

    def temporal_split_losses(
        self, loss_values, temporal_loss_intervals, loss_name, dset_name, fname="full"
    ):
        new_losses = {}
        # Average over time interval
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_values.mean()
        # Don't compute sublosses if we only have one interval
        if len(temporal_loss_intervals) == 2:
            return new_losses
        # Break it down by time interval
        for k in range(len(temporal_loss_intervals) - 1):
            start_ind = temporal_loss_intervals[k]
            end_ind = temporal_loss_intervals[k + 1]
            time_str = f"{start_ind}:{end_ind}"
            loss_subset = loss_values[start_ind:end_ind].mean()
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={time_str}"] = loss_subset
        return new_losses

    def split_up_losses(
        self, loss_values, loss_name, dset_name, field_names
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        new_losses: dict[str, Any] = {}
        time_logs: dict[str, Any] = {}
        time_steps = loss_values.shape[0]  # we already average over batch
        num_time_intervals = min(time_steps, self.num_time_intervals)
        temporal_loss_intervals = np.linspace(0, np.log(time_steps), num_time_intervals)
        temporal_loss_intervals = [0] + [
            int(np.exp(x)) for x in temporal_loss_intervals
        ]
        # Split up losses by field
        for i, fname in enumerate(field_names):
            time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                :, i
            ].cpu()
            new_losses |= self.temporal_split_losses(
                loss_values[:, i], temporal_loss_intervals, loss_name, dset_name, fname
            )
        # Compute average over all fields
        new_losses |= self.temporal_split_losses(
            loss_values.mean(1), temporal_loss_intervals, loss_name, dset_name, "full"
        )
        time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(1).cpu()
        return new_losses, time_logs

    @torch.no_grad()
    def validation_loop(
        self,
        dataloaders: list[WellDataset],
        valid_or_test: str = "valid",
        full=False,
        epoch: int = 0,
    ) -> tuple[float, dict[str, Any]]:
        """Run validation by looping over the dataloader."""
        # Validate the same dataset over FSDP groups but do whatever over DDP groups
        self.model.eval()
        validation_loss = 0.0
        loss_dict: dict[str, Any] = {}
        time_logs: dict[str, Any] = {}
        plot_dicts: dict[str, Any] = {}
        metadatas = []
        collected_ds = set()
        # dataloasorted(dataloaders, key=lambda x: -len(x))
        # dataloaders = pad_validation_data(dataloaders, self.short_validation_length if not full else None)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.rank == 0:
            os.makedirs(os.path.join(self.viz_folder, "loss_names_temp"), exist_ok=True)
        # Each dataset being validated gets separate loader
        for i, dataloader in enumerate(dataloaders):
            # Grab metadata for the current dataset
            assert (
                len(dataloader.dataset.sub_dsets) == 1
            ), "Only one dataset per validation dataloader"
            dataset = dataloader.dataset.sub_dsets[
                0
            ]  # There is only one dset by design
            current_metadata = dataset.metadata
            if current_metadata.dataset_name not in collected_ds:
                metadatas.append(current_metadata)
                collected_ds.add(current_metadata.dataset_name)
            dset_name = current_metadata.dataset_name
            field_names = flatten_field_names(current_metadata, include_constants=False)
            inner_loss_dict = {}
            # NOTE MM - We want each FSDP group to process a different dataset to accelerate 
            # partial validation between epochs. I ran into a lot of issues trying to get this to work
            # without each job at least constructing the loss object in the same
            # way so for now the way this works is that nodes that aren't "assigned" the dataset
            # are given dummy values that they compute the loss over while
            # the assigned nodes actually run the model forward.
            # A better option would be to precompute the total number of fields
            # and build one loss tensor ahead of time that does this, but 
            # this requires more work than I have time for right now.
            rank_assignment = i % self.num_sync_groups # Use same data across one sync group
            # Only print if we're doing something on this node
            if rank_assignment == self.sync_group_rank:
                logger.info(
                    f"Validating dataset {dataset.metadata.dataset_name} with full_trajectory_mode={dataset.full_trajectory_mode} on rank {self.rank}"
                )
            count = 0
            denom = (len(dataloader) if full else min(len(dataloader), self.short_validation_length))
            
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.enable_amp,
                dtype=self.amp_type,
            ):
                for j, batch in enumerate(dataloader):
                    if rank_assignment == self.sync_group_rank:
                        logger.info(f"Starting {valid_or_test}: {dset_name}, Batch {j+1}/{denom}, Rank {self.rank:>3}")
                    
                    count += 1
                    if not full and count >= self.short_validation_length:
                        break

                    start_time = time.time()
                    # Rollout for length of target
                    y_pred, y_ref = self.rollout_model(self.model, batch, self.formatter_dict[dset_name], 
                                                       train=False, fake_pass=(rank_assignment != self.sync_group_rank))
                    assert (y_ref.shape == y_pred.shape), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                
                    if self.save_raw_predictions and rank_assignment == self.sync_group_rank:
                        save_dir = self.experiment_folder+"raw_predictions/"+dset_name+"/"
                        pred_path = save_dir+self.experiment_name+"_pred.npy"
                        ref_path = save_dir+self.experiment_name+"_ref.npy"
                        os.makedirs(save_dir, exist_ok=True)
                        np.save(pred_path, y_pred.cpu().numpy())
                        np.save(ref_path, y_ref.cpu().numpy())

                # Go through losses
                model_time = time.time() - start_time
                y_pred, y_ref = y_pred[..., batch["field_mask"]], y_ref[..., batch["field_mask"]]
                used_field_names = [f for i, f in enumerate(field_names) if batch["field_mask"][i]]
                # used_field_names = field_names
                for loss_fn in self.validation_suite:
                    # Mean over batch and time per field
                    loss = loss_fn(y_pred, y_ref, current_metadata)
                    # Some losses return multiple values for efficiency
                    if not isinstance(loss, dict):
                        loss = {loss_fn.__class__.__name__: loss}
                    # Split the losses and update the logging dictionary
                    for k, v in loss.items():
                        sub_loss = v.mean(0)  # Take the batch mean
                        new_losses, new_time_logs = self.split_up_losses(
                            sub_loss, k, dset_name, used_field_names
                        )
                        # TODO get better way to include spectral error.
                        if k in long_time_metrics or "spectral_error" in k:
                            time_logs |= new_time_logs
                        for loss_name, loss_value in new_losses.items():
                            inner_loss_dict[loss_name] = (
                                inner_loss_dict.get(loss_name, 0.0) + loss_value / denom
                            )
                            # Let's just store the VRMSE since that's what I'm actually looking at on aggregate.
                            if "full_VRMSE_T=all" in loss_name:
                                vrmse = loss_value.item()
                total_time = time.time() - start_time
                max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
                # Only print out if you're actually doing something
                if rank_assignment == self.sync_group_rank:
                    logger.info(
                        f"{valid_or_test}: {dset_name}, Batch {j+1}/{denom}, Rank {self.rank:>3}: Field-time-averaged VRMSE {vrmse:7.4f}, mem {max_mem_GB:5.2f} GB, total_time {total_time:5.3f}s, model {model_time:5.4f}s"
                    )
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                

            # If there is a FSDP group, average loss over the group
            if self.sync_group:
                for k, v in inner_loss_dict.items():
                    dist.all_reduce(inner_loss_dict[k], op=dist.ReduceOp.AVG, group=self.sync_group)
            # Update the overall loss dict with the new loss
            for k, v in inner_loss_dict.items():
                loss_dict[k] = loss_dict.get(k, 0.0) + inner_loss_dict[k]
                
            # Last batch plots - too much work to combine from batches - will be noisy
            if self.rank_in_sync_group == 0 and rank_assignment == self.sync_group_rank:
                if self.image_validation:
                    for plot_fn in validation_plots:
                        plot_fn(
                            y_pred,
                            y_ref,
                            current_metadata,
                            self.viz_folder,  # Temporary until we port over the resume logic
                            epoch,
                            experiment_name = self.experiment_name,
                        )
                if dataset.full_trajectory_mode:
                    # Only plot if we have more than one timestep, but then track loss over timesteps
                    if self.image_validation:
                        plot_all_time_metrics(
                            time_logs, current_metadata, self.viz_folder, epoch
                        )
                    if self.video_validation and count == denom:  
                        try:
                            make_video(
                                y_pred[0],
                                y_ref[0],
                                current_metadata,
                                self.viz_folder,  # Temporary until we port over the resume logic
                                # Put results folder here
                                epoch,
                                field_name_overrides=used_field_names,
                                size_multiplier=.4,
                                experiment_name = self.experiment_name,
                            )
                        except CalledProcessError as e:
                            logger.warning(
                                f"Error in making video due to FFMPEG: {e}. Skipping video."
                            )
                        
        if self.is_distributed:
            # Wait for all ranks to finish
            logger.debug(f"Rank {self.rank} waiting for barrier")
            dist.barrier()
            logger.debug(f"Rank {self.rank} passed barrier")
            # NOTE - Cleaner to just divide through instead of finding ddp group which may not exist
            for k, v in loss_dict.items():
                dist.all_reduce(loss_dict[k], op=dist.ReduceOp.SUM,) 
                loss_dict[k] = loss_dict[k] / self.sync_group_size
        # Single score validation loss is average of all losses on the training metric
        validation_loss = sum(
            [
                loss_dict[
                    f"{metadata.dataset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
                ].item()
                for metadata in metadatas
            ]
        ) / len(metadatas)
        loss_dict = {f"{valid_or_test}_{k}": v.item() for k, v in loss_dict.items()}
        loss_dict |= plot_dicts
        # Misc metrics
        loss_dict["param_norm"] = param_norm(self.model.parameters())
        return validation_loss, loss_dict

    def train_one_epoch(
        self, epoch: int, dataloader: DataLoader
    ) -> tuple[float, dict[str, Any]]:
        """Train the model for one epoch by looping over the dataloader."""
        self.model.train()
        epoch_loss = 0.0
        avg_grad_norm = torch.tensor(0.0).to(self.device)
        last_grad_norm = torch.tensor(0.0).to(self.device)
        train_logs: dict[str, Any] = {}
        batch_start = time.time()
        interval_start = time.time()
        # When using grad acculuation, it makes sense to zero gradient outside first, then after optimizer step
        self.optimizer.zero_grad()  # Set to none now default
        with nullcontext():
            for i, batch in enumerate(dataloader):
                batch["field_mask"] = batch["field_mask"].to(self.device, non_blocking=True)
                # Update grad if we're not using distribution
                update_grad = (i + 1) % self.grad_acc_steps == 0
                with record_function("train batch"):
                    with nullcontext() if (
                        update_grad or self.distribution_type == "local"
                    ) else self.model.no_sync():
                        with torch.autocast(
                            device_type=self.device.type,
                            enabled=self.enable_amp,
                            dtype=self.amp_type,
                        ):
                            data_time = time.time() - batch_start
                            current_metadata = batch["metadata"]
                            dset_name = current_metadata.dataset_name
                            with record_function("model forward"):
                                # (B, T, H, [W, D], C), (B, T, H, [W, D], C)
                                y_pred, y_ref = self.rollout_model(
                                    self.model, batch, self.formatter_dict[dset_name]
                                )
                            if y_pred.shape[1] > self.minimum_context:
                                y_ref = y_ref[:, self.minimum_context:]
                                y_pred = y_pred[:, self.minimum_context:]
                            forward_time = time.time() - batch_start - data_time
                            with record_function("loss computation"):
                                assert (
                                    y_ref.shape == y_pred.shape
                                ), f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                                loss = (
                                    self.loss_multiplier
                                    * self.loss_fn(y_pred, y_ref, current_metadata).mean()
                                    / self.grad_acc_steps
                                )
                                del y_pred, y_ref  # Free up a little before the BW pass
                        with record_function("backward pass"):
                            self.grad_scaler.scale(loss).backward()
                            backward_time = time.time() - batch_start - forward_time - data_time
                    # On update_grad steps, we actually perform the steps
                    with record_function("optimization"):
                        if update_grad:
                            with record_function("clip grad"):
                                if self.clip_gradient > 0 or self.gradient_log_level > 0:
                                    self.grad_scaler.unscale_(self.optimizer)
                                if self.gradient_log_level == 1:
                                    if hasattr(self.model, "sharding_strategy"):
                                        last_grad_norm = get_grad_norm_fsdp(
                                            self.model,
                                            self.rank,
                                            self.world_size,
                                            self.model.sharding_strategy,
                                        )
                                    else:
                                        last_grad_norm = get_grad_norm_local(self.model)
                                        avg_grad_norm += last_grad_norm.detach() / (
                                            len(dataloader) / self.grad_acc_steps
                                        )
                                if self.clip_gradient > 0:
                                    if hasattr(self.model, "clip_grad_norm_"):
                                        self.model.clip_grad_norm_(
                                            self.clip_gradient,
                                            norm_type=2.0,
                                        )
                                    else:
                                        torch.nn.utils.clip_grad_norm_(
                                            self.model.parameters(), self.clip_gradient, norm_type=2.0
                                        )
                            with record_function("actual opt step"):
                                self.grad_scaler.step(self.optimizer)
                                self.grad_scaler.update()
                                self.optimizer.zero_grad()  # Set to none is now default\
                                if self.lr_scheduler_per_step and self.lr_scheduler:
                                    self.lr_scheduler.step()
                    total_time = time.time() - batch_start
                    optimizer_time = total_time - forward_time - backward_time - data_time
                    # Syncing for all reduce anyway so may as well compute synchornous metrics
                    epoch_loss += (self.grad_acc_steps * loss.detach()) / len(
                        dataloader
                    )  # Unscale loss for accurate measure.

                    max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
                    with record_function("logging"):
                        if i % self.log_interval == 0:
                            timing = (time.time() - interval_start) / self.log_interval
                            interval_start = time.time()
                            logger.info(
                                f"Epoch {epoch:>4}, Batch {i+1}/{len(dataloader)}, Rank {self.rank:>3}, SyncStep: {update_grad}:\n\t Data: {current_metadata.dataset_name:<32}, loss {(self.grad_acc_steps*loss.item())**.5:7.4f}, mem {max_mem_GB:5.2f} GB, total_time {timing:5.3f}s, data {data_time:5.4f}s, fwd {forward_time:5.3f}s, bw {backward_time:5.3f}s, opt {optimizer_time:5.3f}s"
                            )
                            # Log times and memory stats to wandb - I don't trust wandb numbers
                            if torch.cuda.is_available():
                                torch.cuda.reset_peak_memory_stats()
                        # Log times and memory stats to wandb - I don't trust wandb numbers
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                        batch_start = time.time()
                        # Log elapsed times in train_log - NOTE: only accurate if cuda syncing, but can be interpretted either way
                        train_logs["avg_data_loading_time"] = train_logs.get(
                            "data_loading_time", 0
                        ) + data_time / len(dataloader)
                        train_logs["avg_forward_time"] = train_logs.get(
                            "forward_time", 0
                        ) + forward_time / len(dataloader)
                        train_logs["avg_backward_time"] = train_logs.get(
                            "backward_time", 0
                        ) + backward_time / len(dataloader)
                        train_logs["avg_optimizer_time"] = train_logs.get(
                            "optimizer_time", 0
                        ) + optimizer_time / len(dataloader)
                        train_logs["avg_time_per_step"] = train_logs.get(
                            "avg_time_per_step", 0
                        ) + total_time / len(dataloader)
                        train_logs["peak_memory"] = max(
                            train_logs.get("peak_memory", 0), max_mem_GB
                        )
                        del batch # Free up memory
                        # prof.step()
            with record_function("finish logging"):
                train_logs["train_loss"] = epoch_loss
                if self.gradient_log_level >= 1:
                    train_logs["avg_grad_norm"] = avg_grad_norm.item()
                if self.lr_scheduler:
                    if not self.lr_scheduler_per_step:
                        self.lr_scheduler.step()
                    train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs

    def validate_if_necessary(
        self,
        epoch: int,
        one_step_dataloaders: list[DataLoader],
        rollout_dataloaders: list[DataLoader],
        valid_or_test: Literal["valid", "test"] = "valid",
    ):
        """Check what type of validate/rollouts we need to do for a given epoch.

        Parameters
        ----------
        epoch: int
            The current epoch. Used for logging and saving checkpoints.
        one_step_dataloaders: list[DataLoader]
            List of dataloaders for one step validation
        rollout_dataloaders: list[DataLoader]
            List of dataloaders for rollout validation
        valid_or_test: str
            String to indicate if we are validating or testing. Options are "valid" or "test"
        """
        is_test = valid_or_test == "test"  # Check if test
        val_loss, rollout_val_loss = None, None
        # First do one step checks = frequency, last epoch, or test. Only do full validation on last epoch or test
        if epoch % self.val_frequency == 0 or epoch == self.max_epoch or is_test:
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}, rank {self.rank}: starting {valid_or_test} validation"
            )
            val_loss, loss_dict = self.validation_loop(
                one_step_dataloaders,
                valid_or_test=valid_or_test,
                full=(epoch == self.max_epoch or is_test),
                epoch=epoch,
            )
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}, rank {self.rank} : {valid_or_test} loss {val_loss}"
            )
            loss_dict |= {f"{valid_or_test}": val_loss, "epoch": epoch}

            if self.wandb_logging and self.rank == 0:
                wandb.log(loss_dict)

        # Rollout if frequency, last epoch, or if this is the test set
        if (
            epoch % self.rollout_val_frequency == 0
            or epoch == self.max_epoch
            or is_test
        ):
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}, rank {self.rank}: starting rollout {valid_or_test} validation"
            )
            rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                rollout_dataloaders,
                valid_or_test=f"rollout_{valid_or_test}",
                full=epoch == self.max_epoch,
                epoch=epoch,
            )
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}, rank {self.rank}: rollout {valid_or_test} loss {rollout_val_loss}"
            )
            rollout_val_loss_dict |= {
                f"rollout_{valid_or_test}": rollout_val_loss,
                "epoch": epoch,
            }
            if self.wandb_logging and self.rank == 0:
                wandb.log(rollout_val_loss_dict)
        return val_loss, rollout_val_loss

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        checkpoint_future = None
        val_loss = self.start_val_loss
        train_dataloader = self.datamodule.train_dataloader(self.sampling_rank)
        test_dataloaders = self.datamodule.test_dataloaders()
        rollout_test_dataloaders = self.datamodule.rollout_test_dataloaders()
        for epoch in range(
            self.start_epoch, self.max_epoch + 1
        ):  # I like 1 indexing for epochs
            # NOTE - only update train sampler because we want to sample same valid data every time
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            # Empty mem caches before train loop
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")

            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            # prof.export_chrome_trace("trace.json")
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: training loss {train_loss:.4f}"
            )
            train_logs |= {"train": train_loss, "epoch": epoch}
            if self.wandb_logging and self.rank == 0:
                wandb.log(train_logs)
            # Empty mem caches before val
            torch.cuda.empty_cache()
            gc.collect()
            # Recreate loader every time so we're using same data in val
            val_dataloders = self.datamodule.val_dataloaders(replicas=self.sync_group_size, 
                                                             rank=self.rank_in_sync_group,
                                                             full=(epoch == self.max_epoch))
            rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders(replicas=self.sync_group_size,
                                                                              rank=self.rank_in_sync_group)
            maybe_val_loss, rollout_loss = self.validate_if_necessary(
                epoch, val_dataloders, rollout_val_dataloaders
            )
            val_loss = maybe_val_loss if maybe_val_loss is not None else val_loss
            if checkpoint_future is not None:
                logger.debug(
                    f"Wait for previous checkpointing {checkpoint_future} to complete."
                )
                checkpoint_future.result()  # Make sure previous checkpoint has finished before starting next.
            # Save "last" every epoch plus various intervals/best results
            checkpoint_future = self.save_model_if_necessary(
                epoch, val_loss, last=(epoch == self.max_epoch)
            )
        # Do test validation
        self.validate_if_necessary(
            epoch, test_dataloaders, rollout_test_dataloaders, valid_or_test="test"
        )

    def validate(self, short_validation=False):
        """Run validation and test. This is a stand alone path"""
        val_dataloders = self.datamodule.val_dataloaders()
        rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders()
        test_dataloaders = self.datamodule.test_dataloaders()
        rollout_test_dataloaders = self.datamodule.rollout_test_dataloaders()

        #Run validation and test                            
        if short_validation == True:
                self.validate_if_necessary(self.max_epoch + 1,
                                           val_dataloders,               
                                           rollout_test_dataloaders,
                                           valid_or_test="valid",) 
        else:
            self.validate_if_necessary(self.max_epoch + 1,
                                    val_dataloders,
                                    rollout_val_dataloaders)
            
            self.validate_if_necessary(self.max_epoch + 1,
                                    test_dataloaders, 
                                    rollout_test_dataloaders,
                                    valid_or_test="test",)

