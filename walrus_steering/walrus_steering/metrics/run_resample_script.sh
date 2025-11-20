#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH -p polymathic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -C ib-genoa
#SBATCH --exclusive

source /mnt/home/mmccabe/venvs/well_venv/bin/activate

srun python /mnt/home/mmccabe/projects/the_well/the_well/benchmark/metrics/resample_mhd.py $1