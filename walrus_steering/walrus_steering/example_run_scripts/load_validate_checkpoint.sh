#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH -J shear-hmlp
#SBATCH --exclude=workergpu006,workergpu094
##SBATCH --nodelist=workergpu057,workergpu058
#SBATCH --output=Rio-%j-%N.log

# Environment Variables
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
##export NCCL_IB_DISABLE=1  # Disable InfiniBand if causing issues
##export NCCL_SOCKET_IFNAME=eno1  # Set correct network interface
##export NCCL_P2P_DISABLE=0
##export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set Rendezvous Endpoint
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
source /mnt/home/pmukhopadhyay/projects/multiple_physics_pretraining/myenv3.10/bin/activate
export PYTHONPATH=$PYTHONPATH:/mnt/home/pmukhopadhyay/projects/temporary_mppx_name/

echo "PYTHONPATH: $PYTHONPATH"
echo "Which Python: $(which python)"
echo "Python Version: $(python --version)"
echo "Python sys.path:"
python -c "import sys; print('\n'.join(sys.path))"

srun python -u `which torchrun` \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=$SLURM_GPUS_PER_NODE \
	--rdzv_id=$SLURM_JOB_ID \
		--rdzv_backend=c10d \
		--rdzv_endpoint=$SLURMD_NODENAME:29500 \
		train.py distribution=local name=Walrus_attempt3_DiffEnc_1_3B validation_mode=True trainer.grad_acc_steps=4 server=rusty optimizer=adam optimizer.lr=3.3e-4 logger.wandb_project_name="MPPX_Training_Attempts" \
			trainer.enable_amp=False model.gradient_checkpointing_freq=4 trainer.log_interval=200 trainer.clip_gradient=10 data.module_parameters.batch_size=10 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 model/processor/space_mixing=full_spatial_attention  \
			model.projection_dim=48 model.intermediate_dim=352 model.hidden_dim=1408 model.groups=16 model.processor_blocks=40 model.drop_path=0.1 \
			model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.num_heads=16 model.processor.time_mixing.num_heads=16 \
			model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 trainer.short_validation_length=40 \
			lr_scheduler=inv_sqrt_w_sqrt_ramps trainer.val_frequency=2 trainer.rollout_val_frequency=5 data.module_parameters.max_dt_stride=5 \
			trainer.prediction_type="delta" data=walrus_test trainer.max_epoch=220 data_workers=10 auto_resume=False