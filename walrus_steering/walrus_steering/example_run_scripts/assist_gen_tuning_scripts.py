# ruff: noqa: E501

from typing import Any, Dict, List

options: Dict[str, List[Any]] = {
    "lr": [0.001, 0.0001, 0.01],
    "causal_in_time": [False, True],
    "prediction_type": ['"delta"', '"full"'],
    "jitter_patches": [True, False],
    "space_mixing": ["full_spatial_attention", "axial_spatial_attention"],
}

SLURM_CPUS_ON_NODE = "{SLURM_CPUS_ON_NODE}"

count = 0
for lr in options["lr"]:
    for prediction_type in options["prediction_type"]:
        for causal_in_time in options["causal_in_time"]:
            for jitter_patches in options["jitter_patches"]:
                for space_mixing in options["space_mixing"]:
                    print(lr, prediction_type, causal_in_time, jitter_patches)

                    base_tuning_string = f"""#!/bin/bash -l
#SBATCH --time=72:00:00
#SBATCH -p gpuxl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=24
#SBATCH -J MPPX_WELL_DEBUGGING
#SBATCH --output=training-%j.log
#SBATCH -C h100

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
# export NCCL_DEBUG=TRACE

# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
source /mnt/home/mmccabe/venvs/well_venv/bin/activate

# Launch the training script

srun python `which torchrun` \\
\t--nnodes=$SLURM_JOB_NUM_NODES \\
\t--nproc_per_node=$SLURM_GPUS_PER_NODE \\
\t--rdzv_id=$SLURM_JOB_ID \\
\t\t--rdzv_backend=c10d \\
\t\t--rdzv_endpoint=$SLURMD_NODENAME:29500 \\
\t\ttrain.py distribution=fsdp server=gpuxl optimizer.lr={lr} logger.wandb_project_name="MPPX_Tuning_1B" \\
\t\t\tdata.module_parameters.batch_size=1 model.hidden_dim=1408 model.groups=16 model.processor_blocks=36 model.drop_path=.1 \\
\t\t\tmodel/processor/space_mixing={space_mixing} model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \\
\t\t\tmodel.causal_in_time={causal_in_time} model.jitter_patches={jitter_patches} \\
\t\t\ttrainer.prediction_type={prediction_type} data=all_2d trainer.max_epoch=200 data_workers=8 auto_resume=False
"""
                    with open(f"tuning_script_{count}.sh", "w") as f:
                        f.write(base_tuning_string)
                    count += 1
