"""Directly inspired from https://pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html"""

from __future__ import annotations

import logging
import pathlib
import shutil
import warnings
from concurrent.futures import Future
from typing import Callable, Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from walrus_steering.optim.distributed_shampoo.distributed_shampoo import (
    DistributedShampoo,
)

logger = logging.getLogger(__name__)

CHECKPOINT_METADATA_FILENAME = "metadata.pt"

def delete_folder_or_symlink(folder: pathlib.Path):
    """Delete a folder or a symlink to a folder."""
    if folder.is_symlink():
        folder.unlink()
    else:
        shutil.rmtree(folder)


def checkpoint_already_exists(checkpoint_dirname: pathlib.Path) -> bool:
    """Check if a given checkpoint already exists."""
    return checkpoint_dirname.exists()


def link_checkpoint(src_checkpoint: pathlib.Path, target_checkpoint: pathlib.Path):
    """Create a symbolic link to an already existing checkpoint.
    The link points to `src_checkpoint` and is named `target_checkpoint`.
    It allows avoiding expensive copies of checkpoints when they refer to the same data.
    To be used typically for saving last checkpoint that refers to an already existing one.
    """
    # Link already exists
    logger.info(f"Link checkpoint {target_checkpoint} to {src_checkpoint}")
    if target_checkpoint.exists() and target_checkpoint.is_symlink():
        target_checkpoint.unlink()
    elif target_checkpoint.exists():
        shutil.rmtree(target_checkpoint)
    target_checkpoint.symlink_to(src_checkpoint, target_is_directory=True)


def save_metadata(
    checkpoint_dir: pathlib.Path,
    epoch: Optional[int] = None,
    val_loss: Optional[float] = None,
    best_val_loss: Optional[float] = None,
):
    """Checkpoint information that do not require synchronization across GPUs,
    or which are already synchronized.
    To be used in combination of FSDP checkpointing strategy.

    """
    state_dict = {"epoch": epoch, "val_loss": val_loss, "best_val_loss": best_val_loss}
    torch.save(state_dict, checkpoint_dir / CHECKPOINT_METADATA_FILENAME)


def on_future(fn: Callable, *args, **kwargs) -> Callable:
    """Wrap any callable to be run as future callback."""

    def future_wrapper(future: Future):
        return fn(*args, **kwargs)

    return future_wrapper


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(
        self,
        model,
        optimizer=None,
    ):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        if isinstance(self.optimizer, DistributedShampoo):
            model_state_dict, _ = get_state_dict(self.model, [])
            optimizer_state_dict = self.optimizer.distributed_state_dict(
                key_to_param=self.model.named_parameters()
            )
        else:
            model_state_dict, optimizer_state_dict = get_state_dict(
                self.model, self.optimizer
            )
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        if isinstance(self.optimizer, DistributedShampoo):
            set_state_dict(
                model=self.model,
                optimizers=[],
                model_state_dict=state_dict["model"],
                optim_state_dict=None,
                options=StateDictOptions(strict=False),
            )
            self.optimizer.load_distributed_state_dict(
                state_dict["optimizer"], key_to_param=self.model.named_parameters()
            )
        else:
            set_state_dict(
                model=self.model,
                optimizers=self.optimizer,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optimizer"],
                options=StateDictOptions(strict=False),
            )


class CheckPointLoader:
    """Base class for checkpointing.
    It only load checkpoints.

    """

    def __init__(
        self,
        save_dir: pathlib.Path | str,
        rank=0,
        load_checkpoint_path: Optional[pathlib.Path | str] = None,
    ) -> None:
        self.save_dir = pathlib.Path(save_dir).resolve()
        self.load_checkpoint_path = (
            pathlib.Path(load_checkpoint_path).resolve()
            if load_checkpoint_path is not None
            else self.last_checkpoint
        )
        self.rank = rank
        self._best_metrics: Optional[float] = None

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        """Load the model and optimizer state from a checkpoint."""

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = self.load_checkpoint_path / "coalesced.pth"
        metadata_path = self.load_checkpoint_path / "metadata.pt"
        epoch, val_loss = None, None

        if checkpoint_path.exists():
            print(f"Loading collapsed checkpoint from {checkpoint_path} onto {device}")
            # Load checkpoint from file
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if "app" in checkpoint:
                app_state_dict = checkpoint["app"]
                if not isinstance(app_state_dict, dict):
                    raise TypeError(f"Expected 'app' key to contain a dict, got {type(app_state_dict)} instead.")

                # Load model state if available
                if "model" in app_state_dict:
                    try:
                        model.load_state_dict(app_state_dict["model"])
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to load model state. Error: {e}")
                else:
                    print("⚠️ Warning: 'model' key is missing in checkpoint. Model weights not restored.")

                # Load optimizer state if available
                if "optimizer" in app_state_dict and optimizer is not None:
                    try:
                        optimizer.load_state_dict(app_state_dict["optimizer"])
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to load optimizer state. Error: {e}")
                else:
                    print("⚠️ Warning: 'optimizer' key is missing in checkpoint. Optimizer state not restored.")

            else:
                raise KeyError(f"Expected 'app' key in {checkpoint_path}, but found: {checkpoint.keys()}")

            # Load metadata if available
            if metadata_path.exists():
                checkpoint_metadata = torch.load(metadata_path, map_location=device)
                epoch = checkpoint_metadata.get("epoch", None)
                val_loss = checkpoint_metadata.get("val_loss", None)
                self._best_metrics = checkpoint_metadata.get("best_val_loss", None)
            else:
                epoch, val_loss = None, None

        else:
            # Fall back to original DCP loading
            state_dict = {"app": AppState(model, optimizer)}

            if self.load_checkpoint_path.exists():
                metadata_file = self.load_checkpoint_path / CHECKPOINT_METADATA_FILENAME
                assert metadata_file.exists(), f"Metadata file {metadata_file} does not exist"

                try:
                    dcp.load(state_dict=state_dict, checkpoint_id=self.load_checkpoint_path)
                    checkpoint_metadata = torch.load(metadata_file, map_location=device)
                except Exception as e:
                    raise RuntimeError(f"Failed to load distributed checkpoint: {e}")
                else:
                    epoch = checkpoint_metadata.get("epoch", None)
                    val_loss = checkpoint_metadata.get("val_loss", None)
                    self._best_metrics = checkpoint_metadata.get("best_val_loss", None)

        return epoch, val_loss

    def save_if_necessary(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        force: bool = False,
    ):
        pass

    @property
    def last_checkpoint(self) -> pathlib.Path | None:
        """Return the real path of the last checkpoints in the directory."""
        last_checkpoint_dir = pathlib.Path(self.save_dir).joinpath("last")
        if last_checkpoint_dir.exists():
            return last_checkpoint_dir.resolve()
        else:
            warnings.warn("No last checkpoint found")
            return None


class CheckPointer(CheckPointLoader):
    """Class to checkpoint training state_dict under FSDP strategy."""

    def __init__(
        self,
        save_dir: pathlib.Path,
        save_best: bool = True,
        checkpoint_frequency: int = 0,
        rank: int = 0,
        load_checkpoint_path: Optional[pathlib.Path] = None,
    ):
        super().__init__(save_dir, rank, load_checkpoint_path)
        logger.info(f"Checkpointing to {save_dir}")
        if rank == 0:
            self.save_dir.mkdir(exist_ok=True)
        self.save_best = save_best
        self.checkpoint_frequency = checkpoint_frequency

    def save_if_necessary(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        val_loss: Optional[float] = None,
        epoch: Optional[int] = None,
        force: bool = False,
    ) -> Optional[Future]:
        """Check if checkpoints must be saved.
        If so triggers the asynchronous writing of the file.
        Upon writing link additional checkpoints to the written file.
        Returns an optional future.

        Args:
            model: Model whose state is to save
        optimizer: Optimizer state to save
        val_loss: Loss saved as checkpoint metadata and used to save best model
        epoch: Epoch saved as checkpoint metadata and used to save the model every n epochs
        force: Boolean flag to force saving the model, preempts saving every n epoch behavior
        """
        state_dict = {"app": AppState(model, optimizer)}
        checkpoint_future = None
        checkpoint_dirnames = []
        # Save checkpoint based on epoch
        # Those checkpoints as they are absolute (comparatively to best) should be the ones saved
        # Other checkpoints can refer to these ones and thus be linked to avoid hard copies
        save_this_epoch = False
        epoch_checkpoint_dirname = self.save_dir / f"step_{epoch}"
        # Force saving checkpoint which does not already exists
        # Typically occurs when saving checkpoint at the end of training
        if checkpoint_already_exists(epoch_checkpoint_dirname):
            save_this_epoch = False
        elif force:
            save_this_epoch = True
        # Epoch number triggers checkpointing
        elif (
            epoch is not None
            and self.checkpoint_frequency
            and (epoch % self.checkpoint_frequency) == 0
        ):
            save_this_epoch = True
        if save_this_epoch:
            checkpoint_dirnames.append(epoch_checkpoint_dirname)

        # Best metrics triggers checkpointing
        if self.save_best:
            assert (
                val_loss is not None
            ), "Expect to save best metrics but no metrics provided."
            if self._best_metrics is None or val_loss < self._best_metrics:
                self._best_metrics = val_loss
                checkpoint_dirnames.append(self.save_dir / "best")

        # Several files should be saved for the same checkpoints
        checkpoint_dirnames.append(self.save_dir / "last") # This is supposed to write every "epoch"
        if checkpoint_dirnames:
            actual_checkpoint_dirname = checkpoint_dirnames[0]
            # Something weird going on with "last" checkpoint in torch 2.5
            # if torch.distributed.is_initialized():
            #     torch.distributed.barrier()
            #     if actual_checkpoint_dirname.exists() and self.rank == 0:
            #         delete_folder_or_symlink(actual_checkpoint_dirname)
            #     torch.distributed.barrier()
            # else:
            #     if actual_checkpoint_dirname.exists():
            #         delete_folder_or_symlink(actual_checkpoint_dirname)

            # Only save the first checkpoints
            logger.info(f"Save checkpoint {actual_checkpoint_dirname}")
            checkpoint_future = dcp.async_save(
                state_dict, checkpoint_id=actual_checkpoint_dirname
            )
            # Save already synchronized data on rank 0 only
            # To be used typically for saving epoch and loss
            if self.rank == 0:
                checkpoint_future.add_done_callback(
                    on_future(
                        save_metadata,
                        checkpoint_dir=actual_checkpoint_dirname,
                        epoch=epoch,
                        val_loss=val_loss,
                        best_val_loss=self._best_metrics,
                    )
                )
            # Link the other checkpoints to the one that has been saved
            # Only to be performed once, hence on rank 0
            if self.rank == 0:
                for linked_checkpoint_filename in checkpoint_dirnames[1:]:
                    checkpoint_future.add_done_callback(
                        on_future(
                            link_checkpoint,
                            actual_checkpoint_dirname,
                            linked_checkpoint_filename,
                        )
                    )

        if force and checkpoint_future is not None:
            # Make checkpoint saving synchronous
            checkpoint_future.result()
            checkpoint_future = None

        return checkpoint_future
