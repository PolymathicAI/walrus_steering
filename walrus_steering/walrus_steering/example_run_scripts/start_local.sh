#!/bin/bash -l
#SBATCH -t 0-2
#SBATCH -p gpu
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1

source /mnt/home/rfear/coding/projects/walrus/temporary_mppx_name/venv/bin/activate

srun --pty python -m torch.distributed.run --nnodes=$SLURM_JOB_NUM_NODES \
									 --nproc_per_node=$SLURM_GPUS_PER_NODE \
									 --rdzv_id=$SLURM_JOB_ID \
									 --rdzv_backend=c10d \
									 --rdzv_endpoint=$SLURMD_NODENAME:29500 \
									 start_local.py
