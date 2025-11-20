import functools
import logging
import os
import pathlib
from typing import Dict, Optional, cast

import pickle
import hydra
import torch
import torch.nn.functional as F

import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torchinfo import summary

# TODO - rewrite this for torchrun
from walrus_steering.data import MixedWellDataModule
from walrus_steering.data.well_to_multi_transformer import (
    ChannelsFirstWithTimeFormatter,
)
from walrus_steering.optim.distributed_shampoo.shampoo_types import (
    FSDPShampooConfig,
    HSDPShampooConfig,
)
from walrus_steering.optim.distributed_shampoo.utils.shampoo_fsdp_utils import (
    compile_fsdp_parameter_metadata,
)
from walrus_steering.trainer.checkpoints import CheckPointLoader
from walrus_steering.trainer.training import Trainer
from walrus_steering.utils.distribution_utils import (
    configure_distribution,
    distribute_model,
)
from walrus_steering.utils.experiment_utils import configure_experiment

logger = logging.getLogger("temporary_mppx_name")
# logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = pathlib.Path(__file__).parent / "configs"
CONFIG_NAME = "config"
CONFIG_PATH = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
assert CONFIG_PATH.is_file(), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")

# import warnings
# warnings.filterwarnings("ignore")

def start_training(
    cfg: DictConfig,
    experiment_name: str,
    experiment_folder: str,
    viz_folder: str,
    is_distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    device_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
):
    """Instantiate the different objects required for training and run the training loop."""

    logger.info(f"Instantiate datamodule {cfg.data.wandb_data_name}")
    datamodule: MixedWellDataModule = instantiate(
        cfg.data.module_parameters,
        world_size=world_size,
        rank=rank,
        data_workers=cfg.data_workers,
        well_base_path=cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
    )
    field_to_index_map = datamodule.train_dataset.field_to_index_map
    # TODO - currently enforcing MPP format, but should allow for other types
    # Retrieve the number of fields used in training
    # from the mapping of field to index
    total_input_fields = max(field_to_index_map.values()) + 1

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_states=total_input_fields,
    )
    if rank == 0:
        summary(model, depth=5)

    logger.info(
        f"Assigning distribution strategy: {cfg.distribution.distribution_type}"
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(local_rank)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model = distribute_model(model, cfg, device_mesh)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    _partial = False
    if "DistributedShampoo" in cfg.optimizer._target_:
        # See doc at https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md
        # For DDP, we use the default cfg.optimizer.distributed_config configuration. Otherwise, we override it.
        distribution_type = cfg.distribution.distribution_type.upper()
        if distribution_type == "LOCAL":
            cfg.optimizer.distributed_config = (
                None  # local distribution does not require any special configuration
            )
        elif distribution_type == "DDP":
            pass
        elif distribution_type == "FSDP":
            distributed_config = FSDPShampooConfig(
                param_to_metadata=compile_fsdp_parameter_metadata(model)
            )
            _partial = True  # Hack due to Hydra limitations
        elif distribution_type == "HSDP":
            logger.warning(
                "HSDP requires torch>2.4.1 (_MeshEnv._get_all_submeshes is not implemented in <=2.4.1). Waiting for a stable release of torch before updating requirements."
            )
            if device_mesh is None:
                raise ValueError("`device_mesh` is required for HSDP")
            distributed_config = HSDPShampooConfig(
                param_to_metadata=compile_fsdp_parameter_metadata(model),
                device_mesh=device_mesh,
                num_trainers_per_group=cfg.optimizer.distributed_config.num_trainers_per_group,
            )
            _partial = True  # Hack due to Hydra limitations
        else:
            raise ValueError(f"Unknown distribution type {distribution_type}")

    if _partial:  # Only for distributed_shampoo
        # Just a hack to instantiate the optimizer with the correct distributed_config parameter
        # (Hydra forces us to do it this way)
        _optimizer: functools.partial = cast(
            functools.partial,
            instantiate(cfg.optimizer, params=model.parameters(), _partial_=_partial),
        )
        optimizer: torch.optim.Optimizer = _optimizer(
            distributed_config=distributed_config
        )
    else:
        optimizer = cast(
            torch.optim.Optimizer,
            instantiate(cfg.optimizer, params=model.parameters(), _partial_=_partial),
        )

    # Set start epoch to 0 before potential retrieval from checkpoint
    start_epoch = 1
    last_epoch = -1  # Default for Pytorch
    val_loss = torch.tensor(float("inf"))
    logger.info(f"Instantiate checkpointer {cfg.checkpoint._target_}")
    checkpointer: CheckPointLoader = instantiate(cfg.checkpoint, rank=rank)
    print(f'{checkpointer.load_checkpoint_path}')

    if hasattr(cfg.checkpoint, "save_dir"):
        last_ckpt_dirname = checkpointer.last_checkpoint
        load_ckpt_dirname = checkpointer.load_checkpoint_path
        if load_ckpt_dirname is not None and os.path.exists(load_ckpt_dirname):
            # Load model and optimizer from checkpoint
            logger.info(f"Resume from checkpoint {load_ckpt_dirname}")
            epoch, val_loss = checkpointer.load(model, optimizer)
            # Ensure initial_lr is set for each parameter group
            for param_group in optimizer.param_groups:
                if "initial_lr" not in param_group:
                    param_group["initial_lr"] = cfg.optimizer.lr

            logger.info(f"Resume from epoch {epoch} with validation loss {val_loss}")
            start_epoch = 1 if epoch is None else epoch + 1
            last_epoch = start_epoch - 1  # Set last_epoch to the last completed epoch
    if hasattr(cfg, "lr_scheduler"):
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")

        if cfg.trainer.lr_scheduler_per_step:
            #NOTE(TM): This ideally should be max_iterations or something. Or T_max if we are to go with pytorch.
            step_mult_factor = cfg.data.module_parameters.max_samples / cfg.trainer.grad_acc_steps
        else:
            step_mult_factor = 1

        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.max_epoch,
            step_mult_factor=step_mult_factor,
            last_epoch=max(-1, last_epoch-1),
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    # Update the config with the newly generated field-to-index map for resuming/knowing what was there
    with open_dict(cfg):
        cfg.data.field_index_map_override = field_to_index_map
    if rank == 0:
        logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
        
    activations = {}
    if hasattr(cfg, "layers_to_hook"):     
        if hasattr(cfg, "inject_tensor_path"):
            with open(cfg.inject_tensor_path, 'rb') as handle:
                input_tensor = pickle.load(handle)
            input_tensor = input_tensor.to(device)

            def calc_norm(tensor, dims=(2,3,4,5)): # was: (2,3,4,5)
                norm = torch.sqrt(torch.sum(tensor**2, dim=dims, keepdim=True))
                return norm

            # Injection tensor shape: [1, 1, 1408, 31, 32, 1]
            def create_norm_injection_hook(input_tensor, strength=cfg.inject_strength, inject_sign=cfg.inject_sign):
                cropping_warning_emitted = False  # warn once per layer
                def norm_injection_hook(module, input, output):
                    nonlocal cropping_warning_emitted
                    inject_tensor = input_tensor
                    output_shape = output[0].shape
                
                    # Initial center crop (ensures inject dims <= output dims)
                    spatial_start = len(output_shape) - 3  # Last 3 are spatial
                    if any(inject_tensor.shape[i] > output_shape[i] for i in range(spatial_start, len(output_shape))):
                        slices = []
                        cropped_dims = []
                        for i, (inj_dim, out_dim) in enumerate(zip(inject_tensor.shape, output_shape)):
                            if i >= spatial_start and inj_dim > out_dim:  # Only crop spatial dims
                                start = (inj_dim - out_dim) // 2
                                slices.append(slice(start, start + out_dim))
                                cropped_dims.append((i, inj_dim, out_dim))
                            else:
                                slices.append(slice(None))
                        inject_tensor = inject_tensor[tuple(slices)]

                        if not cropping_warning_emitted:
                            logger.warning(f"Center-cropped injection tensor (mode '{cfg.inject_spatial_interpolation}'). "
                                           f"Cropped dims: {cropped_dims}")
                            cropping_warning_emitted = True

                    # Options skip: exit if tensors now equal
                    if inject_tensor.shape == output_shape:
                        expanded_injection = inject_tensor

                    # Option A: spatial interpolation, then expansion
                    elif cfg.inject_spatial_interpolation == 'interpol':
                        inject_tensor = inject_tensor.squeeze(0)
                        inject_tensor = F.interpolate(inject_tensor,
                                                    size=(output_shape[3],
                                                          output_shape[4],
                                                          output_shape[5]),
                                                    mode='trilinear',
                                                    align_corners=False)
                        inject_tensor = inject_tensor.unsqueeze(0)
                        expanded_injection = inject_tensor.expand(output_shape[0], 
                                                                output_shape[1], 
                                                                -1, -1, -1, -1)

                    # Option B: padding after expanding batch & channel    
                    elif cfg.inject_spatial_interpolation == 'pad':
                        expanded_injection = inject_tensor.expand(output_shape[0],
                                                                    output_shape[1],
                                                                    -1, -1, -1, -1)
                        inject_shape = expanded_injection.shape
                        spatial_pad = [] # Only pad spatial dims (last 3)
                        for i in range(3):
                            dim_idx = len(output_shape) - 1 - i
                            if inject_shape[dim_idx] < output_shape[dim_idx]:
                                spatial_pad.extend([0, output_shape[dim_idx] - inject_shape[dim_idx]])
                            else:
                                spatial_pad.extend([0, 0])
                        padding = tuple(spatial_pad)
                        expanded_injection = F.pad(expanded_injection, padding, mode='constant', value=0)

                    # Option C: drop spatial dims, avg out spatial variation in other dims
                    elif cfg.inject_spatial_interpolation == 'drop':
                        inject_tensor = inject_tensor.mean(dim=(3, 4, 5), keepdim=True)
                        expanded_injection = inject_tensor.expand(output_shape[0], 
                                                                output_shape[1], 
                                                                -1,
                                                                output_shape[3],
                                                                output_shape[4],
                                                                output_shape[5])                        

                    # Option D: no padding, expansion or dropping of spatial dims  
                    elif cfg.inject_spatial_interpolation == 'none':
                        expanded_injection = inject_tensor

                    # Initial normalisation
                    output_norm = calc_norm(output[0])
                    inject_norm = calc_norm(expanded_injection)

                    # Tensor injection
                    if inject_sign =='pos':
                        scaled_ouput = output[0] + (strength * (expanded_injection * (output_norm/inject_norm)))
                    elif inject_sign =='neg':
                        scaled_ouput = output[0] - (strength * (expanded_injection * (output_norm/inject_norm)))

                    # Post-inject renormalisation
                    # scaled_norm = calc_norm(scaled_ouput)            
                    # rescaled_ouput = scaled_ouput * (output_norm/scaled_norm)
                    
                    return (scaled_ouput, *output[1:])
                return norm_injection_hook
            
            layers_to_hook = cfg.get("layers_to_hook", []) 
            for layer_name in layers_to_hook:
                for module_name, module in model.named_modules():
                    if module_name == layer_name:
                        module.register_forward_hook(create_norm_injection_hook(input_tensor))

        else:        
            def get_activations(layer_name):
                def hook(module, input, output):
                    activations[layer_name] = output[0].detach().cpu()
                return hook

            layers_to_hook = cfg.get("layers_to_hook", []) 
            for layer_name in layers_to_hook:
                for module_name, module in model.named_modules():
                    if module_name == layer_name:
                        module.register_forward_hook(get_activations(module_name))
        
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    trainer: Trainer = instantiate(
        cfg.trainer,
        experiment_name=experiment_name,
        viz_folder=viz_folder,
        experiment_folder=experiment_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
        device=device,
        device_mesh=device_mesh,
        distribution_type=cfg.distribution.distribution_type,
        rank=rank,
        world_size=world_size,
        formatter=ChannelsFirstWithTimeFormatter,  # TODO change this to function of model
        wandb_logging=cfg.logger.wandb,
        start_epoch=start_epoch,
        start_val_loss=val_loss,
    )
    if cfg.validation_mode:
        trainer.validate(short_validation=cfg.short_validation_only)
    else:
        # Save config to directory folder
        if rank == 0:
            with open(
                pathlib.Path(experiment_folder) / "extended_config.yaml", "w"
            ) as f:
                OmegaConf.save(cfg, f)
        trainer.train()

    if rank == 0 and cfg.save_activations: 
        activations_dir = "/mnt/home/rfear/coding/projects/walrus/temporary_mppx_name/experiments/activations"
        activations_path = os.path.join(activations_dir, f"{cfg.save_activations}.pickle")
        
        with open(activations_path, 'wb') as f:
            pickle.dump(activations, f)
        logging.info(f"Saved activations to {activations_path}")

    return trainer, model, activations

@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name=str(CONFIG_NAME))

def main(cfg: DictConfig):
    # Torch optimization settings
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    torch.backends.cudnn.allow_tf32 = True
    # Retrieve multiple processes context to setup DDP
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = (cfg.distribution.distribution_type.upper() != "LOCAL" and world_size > 1)

    device_mesh = configure_distribution(cfg)

    (cfg,
    experiment_name,
    experiment_folder,
    checkpoint_folder,
    artifact_folder,
    viz_folder,) = configure_experiment(cfg, rank, is_distributed)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # Initiate wandb logging

    # Make sure we're logging the true batch size
    config_for_wandb = cast(Dict, OmegaConf.to_container(cfg, resolve=True))
    config_for_wandb["world_size"] = world_size

    # Global batch size is microbatch size * number of GPUs * gradient accumulation steps
    config_for_wandb["global_batch_size"] = (cfg.data.module_parameters.batch_size * world_size) * cfg.trainer.grad_acc_steps
    
    if rank == 0 and cfg.logger.wandb:
        wandb.init(
            project=cfg.logger.wandb_project_name,
            group=f"{cfg.data.wandb_data_name}",
            config=config_for_wandb,
            name=experiment_name)

    start_training(cfg,
                   experiment_name,
                   experiment_folder,
                   viz_folder,
                   is_distributed,
                   world_size,
                   rank,
                   local_rank,
                   device_mesh=device_mesh,)
    
    if rank == 0 and cfg.logger.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
