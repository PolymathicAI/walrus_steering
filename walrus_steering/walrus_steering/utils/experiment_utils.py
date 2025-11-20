import logging
import os
import os.path as osp
from typing import Tuple, cast

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)


def configure_paths(experiment_folder, rank=0):
    """Configure the paths for the experiment with the given experiment folder."""
    # Make ____ directory as experiment_folder/______
    if rank == 0:
        os.makedirs(osp.join(experiment_folder, "checkpoints"), exist_ok=True)
        os.makedirs(osp.join(experiment_folder, "artifacts"), exist_ok=True)
        os.makedirs(osp.join(experiment_folder, "viz"), exist_ok=True)
    # Return corresponding paths
    checkpoint_folder = osp.join(experiment_folder, "checkpoints")
    artifact_folder = osp.join(experiment_folder, "artifacts")
    viz_folder = osp.join(experiment_folder, "viz")
    return checkpoint_folder, artifact_folder, viz_folder


def get_experiment_name(cfg: DictConfig) -> str:
    """
    Get the experiment name based on the configuration model, data, and optimizer.

    Used to set default save path if not overridden.

    This is a messy hardcoded process that is likely a good candidate for refactoring.
    """
    # Data section
    data_name = cfg.data.wandb_data_name.replace("_", "")[:5]
    # Model section
    model_name = cfg.model._target_.split(".")[-1].replace("_", "")[:5]
    encoder_name = cfg.model.encoder._target_.split(".")[-1].replace("_", "")[:5]
    decoder_name = cfg.model.decoder._target_.split(".")[-1].replace("_", "")[:5]
    # TODO - this is sloppy but get's what I need for now (getting space/time names). Maybe recursive search for _target_s?
    processor_name = cfg.model.processor._target_.split(".")[-1].replace("_", "")[:5]
    if hasattr(cfg.model.processor, "space_mixing"):
        space_name = cfg.model.processor.space_mixing._target_.split(".")[-1].replace(
            "_", ""
        )[:5]
        processor_name += f"-{space_name}"
    if hasattr(cfg.model.processor, "time_mixing"):
        time_name = cfg.model.processor.time_mixing._target_.split(".")[-1].replace(
            "_", ""
        )[:5]
        processor_name += f"-{time_name}"
    # Optimizer section
    optimizer_name = cfg.optimizer._target_.split(".")[-1].replace("_", "")[:5]
    # Training type section
    prediction_type = cfg.trainer.prediction_type.replace("_", "")[:5]
    aggregate_name = f"{cfg.name}-{data_name}-{prediction_type}-{model_name}[{encoder_name}-{decoder_name}-{processor_name}]-{optimizer_name}-{cfg.optimizer.lr}"
    return aggregate_name


def configure_experiment(
    cfg: DictConfig, rank: int = 0, is_distributed: bool = False
) -> Tuple[DictConfig, str, str, str, str, str]:
    """Works through resume logic to figure out where to save the current experiment
    and where to look to resume or validate previous experiments.

    If the user provides overrides for the folder/checkpoint/config, use them.

    If folder isn't provided, construct default. If autoresume or validation_mode is enabled,
    look for the most recent run under that directory and take the config and weights from it.

    If checkpoint is provided, use it to override any weights obtained until now. If
    any checkpoint is available either in the folder or checkpoint override, this
    is considered a resume run.

    If it's in validation mode but no checkpoint is found, throw an error.

    If config override is provided, use it (with the weights and current output folder).
    Otherwise start search over hierarchy.
      - If checkpoint is being used, look to see if it has an associated config file
      - If no checkpoint but folder, look in folder
      - If not, just use the default config (whatever is currently set)

    Parameters
    ----------
    cfg : DictConfig
        The yaml configuration object being modified/read
    rank : int, optional
        The rank of the current torch process, by default 0
    is_distributed : bool, optional
        Whether the current process is distributed, by default False
    """
    # Sort out default names and folders
    if not cfg.automatic_setup:
        return cfg, cfg.name, ".", "./checkpoints", "./artifacts", "./viz"
    experiment_name = get_experiment_name(cfg)
    if hasattr(cfg, "experiment_dir"):
        base_experiment_folder = cfg.experiment_dir
    else:
        base_experiment_folder = os.getcwd()
    base_experiment_folder = osp.join(base_experiment_folder, experiment_name)
    experiment_folder = cfg.folder_override  # Default is ""
    checkpoint_file = cfg.checkpoint_override  # Default is ""
    config_file = cfg.config_override  # Default is ""
    validation_mode = cfg.validation_mode

    if len(checkpoint_file) > 0:
        raise NotImplementedError("Checkpoint override not yet implemented.")
    # Barrier around this to ensure all processes choose same folder.

    # If using default naming, check for auto-resume, otherwise make a new folder with default name
    if len(experiment_folder) == 0:
        if osp.exists(base_experiment_folder):
            prev_runs = sorted(os.listdir(base_experiment_folder), key=lambda x: int(x))
        else:
            prev_runs = []
        if is_distributed:
            torch.distributed.barrier()
        if (validation_mode or cfg.auto_resume) and len(prev_runs) > 0:
            experiment_folder = osp.join(base_experiment_folder, prev_runs[-1])
        elif validation_mode:
            raise ValueError(
                f"Validation mode enabled but no previous runs found in {base_experiment_folder}."
            )
        else:
            experiment_folder = osp.join(base_experiment_folder, str(len(prev_runs)))
        logger.info(
            f"No override experiment folder detected. Using default experiment folder {experiment_folder}"
        )
    else:
        logger.info(f"Using override experiment folder {experiment_folder}")
    # Barrier around this to ensure all processes choose same folder.
    if is_distributed:
        torch.distributed.barrier()
    if (
        len(config_file) == 0
    ):  # If no config override, check for config file in experiment folder
        config_file = osp.join(experiment_folder, "extended_config.yaml")
        if not osp.isfile(config_file):
            config_file = ""

    ## TODO - once checkpoint override is implemented, this will need to be updated
    # Now check for default checkpoint options - if override used, ignore
    # if osp.exists(experiment_folder) and len(checkpoint_file) == 0:
    #     last_chpt = osp.join(experiment_folder, "checkpoints", "last") # FSDP means these are folders
    #     # If there's a checkpoint file, consider this a resume. Otherwise, this is new run.
    #     if osp.isfile(last_chpt):
    #         checkpoint_file = last_chpt

    # if len(checkpoint_file) > 0:
    #     logger.info(f"Checkpoint found or provided directly, using checkpoint file {checkpoint_file}")
    # if not osp.isfile(checkpoint_file) and len(checkpoint_file) > 0:
    #     raise ValueError(
    #         f"Checkpoint path provided but checkpoint file {checkpoint_file} not found."
    #     )
    # # Now pick a config file to use - either current, override, or related to a different override
    # if len(checkpoint_file) > 0 and len(config_file) == 0:
    #     # Check two levels - the parent folder of the checkpoint and the experiment folder
    #     checkpoint_path = osp.join(
    #         osp.dirname(checkpoint_file), osp.pardir, "extended_config.yaml"
    #     )
    #     folder_path = osp.join(experiment_folder, "extended_config.yaml")
    #     if osp.isfile(checkpoint_path):
    #         logger.info(f"Config file exists relative to checkpoint override provided, \
    #                         using config file {checkpoint_path}")
    #     elif osp.isfile(folder_path):
    #         logger.warn(f"Config file not found in checkpoint override path. \
    #                     Found in experiment folder, using config file {folder_path}. \
    #                     This could lead to weight compatibility issues if the checkpoints do not align with \
    #                     the specified folder.")
    #     else:
    #         logger.warn(
    #             "Checkpoint override provided, but config file not found in checkpoint override path \
    #                     or experiment folder. Using default configuration which may not be compatible with checkpoint."
    #         )
    #     # resume = True
    # elif len(config_file) > 0:
    #     logger.log(f"Config override provided, using config file {config_file}")
    # elif validation_mode:
    #     raise ValueError(
    #         f"Validation mode enabled but no checkpoint provided or found in {experiment_folder}."
    #     )
    if len(config_file) > 0:
        cfg = cast(DictConfig, OmegaConf.load(config_file))

    # Create experiment folder if it doesn't already exist
    if rank == 0:
        os.makedirs(experiment_folder, exist_ok=True)
    checkpoint_folder, artifact_folder, viz_folder = configure_paths(
        experiment_folder, rank=rank
    )
    with open_dict(cfg):
        cfg.checkpoint.save_dir = checkpoint_folder
        cfg.name = experiment_name
    return (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,
        artifact_folder,
        viz_folder,
    )
