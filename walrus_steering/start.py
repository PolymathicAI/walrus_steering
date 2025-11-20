import os
import sys
import logging
import shutil
import imageio_ffmpeg 
import matplotlib as mpl 
from datetime import datetime
from hydra import compose, initialize 
import matplotlib.animation as manimation 
from walrus_steering.train import start_training 
from walrus_steering.utils.distribution_utils import configure_distribution
from walrus_steering.utils.experiment_utils import configure_experiment

def setup_environment():
    # --- FFMPEG Configuration ---
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path


    # --- Setup logging before initializing Hydra ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"experiments/logs/notebook_run_{timestamp}.log"
    os.makedirs("experiments/logs", exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler(sys.stdout)])

def main():
    setup_environment()
                                                                                                 
    # --- Define your run list here --- #
    run_list = [
                # [0.0, "none", "none", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                # [0.0, "none", "none", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1"],
                # [0.1, "pos", "interpol", "euler_multi_quadrants_openBC", "euler_multi_quadrants_openBC_gamma_1.404_H2_100_Dry_air_-15"],
                # [0.1, "neg", "interpol", "euler_multi_quadrants_openBC", "euler_multi_quadrants_openBC_gamma_1.404_H2_100_Dry_air_-15"],
                # [0.1, "pos", "interpol", "euler_multi_quadrants_openBC", "euler_multi_quadrants_openBC_gamma_1.33_H2O_20"],
                # [0.1, "neg", "interpol", "euler_multi_quadrants_openBC", "euler_multi_quadrants_openBC_gamma_1.33_H2O_20"],
                [0.3, "neg", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                [0.3, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                [0.5, "neg", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                [0.5, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                [0.7, "neg", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                [0.7, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1(double-v)"],
                # [0.3, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1"],
                # [0.4, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1"],
                # [0.6, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1"],
                # [0.8, "pos", "drop", "shear_flow", "shear_flow_Reynolds_5e4_Schmidt_5e-1"],
                # [0.1, "pos", "interpol", "rayleigh_benard", "rayleigh_benard_Rayleigh_1e9_Prandtl_10"],
                # [0.1, "neg", "interpol", "rayleigh_benard", "rayleigh_benard_Rayleigh_1e9_Prandtl_10"],
                # [0.1, "pos", "drop", "gray_scott_reaction_diffusion", "gray_scott_reaction_diffusion_gliders_F_0.014_k_0.054"],
                # [0.1, "neg", "drop", "gray_scott_reaction_diffusion", "gray_scott_reaction_diffusion_gliders_F_0.014_k_0.054"],
                # [0.1, "pos", "drop", "gray_scott_reaction_diffusion", "gray_scott_reaction_diffusion_spirals_F_0.018_k_0.051"],
                # [0.1, "neg", "drop", "gray_scott_reaction_diffusion", "gray_scott_reaction_diffusion_spirals_F_0.018_k_0.051"],
                ]

    ## --- Model Loop --- ##
    for i, isign, itype, dset, fname in run_list: 
            
        # --- Experiment Setup --- #
        injection_strength = i
        inject_sign = isign  #     <<--------### ['pos' | 'neg' ] 
        inject_type = itype #      <<--------### ['none' | 'pad' | 'interpol' | 'drop']  
        experiment_name = fname+f"[{inject_sign}-{inject_type}@{injection_strength}][time][FullRes-NoRenorm][euler_single:(dt_stride=2)-(dt_stride=1)]" 
        run_name = f"interpretability[{fname}]"

        experiment_folder = "experiments/"
        viz_folder = "experiments/visuals"
        sv_path = "experiments/checkpointing/"
        os.makedirs(sv_path, exist_ok=True)

        with initialize(version_base=None, config_path="temporary_mppx_name/configs/", job_name=run_name):
            cfg = compose(config_name="config",
                        overrides=[f"name=\"{run_name}\"",
                                    "hydra.run.dir=experiments",
                                    "distribution=local",
                                    "server=rusty",
                                    "logger.wandb=False",
                                    "checkpoint.checkpoint_frequency=0",
                                    "checkpoint.save_best=False",
                                   f"checkpoint.save_dir='experiments/checkpointing/{run_name}/'",
                                    "+checkpoint.load_checkpoint_path='/mnt/home/polymathic/ceph/MPPX_logging/miles_rio_checkpoint_last/'", 
                                    "data.well_base_path='/mnt/home/rfear/coding/projects/walrus/temporary_mppx_name/the_well_internal/datasets/'", # /mnt/home/polymathic/ceph/the_well/datasets/
                                    "data=2d_steering_subset",
                                   f"+data.module_parameters.well_dataset_info.{dset}.include_filters=['{fname}.hdf5']",
                                	"data.module_parameters.batch_size=2",  
                                    "data_workers=1",
                                    "trainer.image_validation=False",          # <<--------###
                                    "trainer.video_validation=True",           # <<--------###
                                    "validation_mode=True",
                                    "short_validation_only=True",
                                    "trainer.val_frequency=300",
                                    "trainer.rollout_val_frequency=1",
                                    "trainer.short_validation_length=2",
                                    "trainer.save_raw_predictions=False",                # <<--------###
                                #   f"trainer.save_activations='{fname}[FullRes]'", # <<--------###
                                    "layers_to_hook=['blocks.39']",#])
                                   f"+inject_spatial_interpolation={inject_type}",
                                   f"+inject_sign={inject_sign}",
                                   f"+inject_strength={injection_strength}",
                                   f"+inject_tensor_path='experiments/activations/newTensor:(18vortex_group)-(10laminar_group).pickle'"]) 


        # --- Distribution --- #
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        is_distributed = (cfg.distribution.distribution_type.upper() != "LOCAL" and world_size > 1)
        device_mesh = configure_distribution(cfg)

        # --- Train --- #
        trainer, model, activations = start_training(cfg=cfg,
                                                    experiment_name=experiment_name,
                                                    experiment_folder=experiment_folder,
                                                    viz_folder=viz_folder,
                                                    is_distributed=is_distributed,
                                                    world_size=world_size,
                                                    local_rank=local_rank,
                                                    rank=rank,
                                                    device_mesh=device_mesh)


if __name__ == "__main__":
    main()