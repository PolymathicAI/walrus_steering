#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH -J MPPX_WELL_DEBUGGING
#SBATCH --output=training-debug-%j.log
#SBATCH -C a100-80gb

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1

# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
source ~/venvs/well_venv/bin/activate

# source /mnt/home/polymathic/ceph/the_well/venv_benchmark_well/bin/activate
# Launch the training script

srun python `which torchrun` \
	--nnodes=$SLURM_JOB_NUM_NODES \
 	--nproc_per_node=$SLURM_GPUS_PER_NODE \
	--rdzv_id=$SLURM_JOB_ID \
		--rdzv_backend=c10d \
		--rdzv_endpoint=$SLURMD_NODENAME:29500 \
		train.py distribution=fsdp server=rusty data=debug trainer=debug data_workers=11 auto_resume=True
