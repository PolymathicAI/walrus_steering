#!/bin/bash
cd temporary_mppx_name/
source /mnt/home/rfear/coding/projects/walrus/temporary_mppx_name/venv/bin/activate

export OMP_NUM_THREADS=24
export HDF5_USE_FILE_LOCKING=FALSE

srun -p gpu -C h100 -t 0-2 --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-gpu=24 \
    --pty python -m torch.distributed.run --nproc_per_node=1 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        start.py
