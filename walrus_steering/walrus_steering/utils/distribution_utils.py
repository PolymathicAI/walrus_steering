import logging
import os

import torch
from hydra.utils import get_class
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

logger = logging.getLogger(__name__)


def configure_distribution(cfg):
    """Configure the appropriate device mesh for the given distribution parameters.

    Parameters:
    ----------
    cfg: DictConfig
        The configuration object for the experiment which contains the appropriate distribution type.

    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    # Initialize the device mesh - right now we have FSDP, HSDP, and DDP - HSDP is the only
    # multi-level sharding strategy

    # NOTE - Documentation suggests you do not need torch.cuda.set_device(local_rank) when using
    # device_mesh, but I got errors until I set it in train.py.
    if cfg.distribution.distribution_type.upper() == "LOCAL":
        return None
    elif cfg.distribution.distribution_type.upper() == "HSDP":
        # HSDP is DDP across nodes, FSDP within nodes - useful if  intra-node xfer speed >>> inter-node xfer speed
        mesh = init_device_mesh(
            "cuda",
            (world_size // local_world_size, local_world_size),
            mesh_dim_names=("ddp", "fsdp"),
        )
    else:
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=(cfg.distribution.distribution_type.lower(),),
        )
    return mesh


def distribute_model(model, cfg, mesh=None):
    """Distributes model over a given mesh using the appropriate distribution strategy.

    Parameters:
    ----------
    model: nn.Module
        The model to be distributed.
    cfg: DictConfig
        The configuration object for the experiment which contains the appropriate distribution type.
    mesh: None | torch.distributed.device_mesh.DeviceMesh
        The device mesh to distribute the model over.
    """
    # If no mesh is provided, there is no valid distribution strategy
    if mesh is None:
        return model
    elif cfg.distribution.distribution_type.upper() in ["FSDP", "HSDP", "DDP"]:
        # FSDP and HSDP both use FSDP API, so need to build in AMP if we're going to use it.
        if cfg.trainer.enable_amp:
            fpSixteen = MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
        else:
            fpSixteen = None

        # TODO - Add in more FSDP parameters from the config including Zero2
        if cfg.distribution.distribution_type.upper() == "FSDP":
            sharding = ShardingStrategy.FULL_SHARD
        elif cfg.distribution.distribution_type.upper() == "HSDP":
            sharding = ShardingStrategy.HYBRID_SHARD
        elif cfg.distribution.distribution_type.upper() == "DDP":
            sharding = ShardingStrategy.NO_SHARD
        # TODO - should make this configurable at some point, but this
        # is a good default for now. Shards per block where block is the encoder, decoder,
        # or processor. For larger enc/dec it probably makes sense to have more blocks.
        wrap_policy = ModuleWrapPolicy(
            [
                # get_class(cfg.model.encoder._target_),
                # get_class(cfg.model.decoder._target_),
                get_class(cfg.model.processor._target_),
            ]
        )
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=fpSixteen,
            sharding_strategy=sharding,
            device_mesh=mesh,
            use_orig_params=False,
        )
    else:
        raise ValueError(
            f"Unknown distribution type {cfg.distribution.distribution_type} - must be LOCAL, DDP, FSDP, or HSDP"
        )
    return model
