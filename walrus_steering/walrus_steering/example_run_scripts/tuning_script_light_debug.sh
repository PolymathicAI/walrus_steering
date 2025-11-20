#!/bin/bash -l
#SBATCH --time=144:00:00
#SBATCH -p gpuxl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=24
#SBATCH -J rb_lower_eps
#SBATCH --output=norm_baselines_debug-%j.log
##SBATCH --dependency=singleton
##SBATCH -C a100-80gb

export OMP_NUM_THREADS=${SLURM_CPUS_ON_NODE}
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
export TORCHELASTIC_ERROR_FILE=/mnt/home/mmccabe/projects/temporary_mppx_name/temporary_mppx_name/torch_worker_log.json
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1
# export CUDA_LAUNCH_BLOCKING=1 # Disable peer-to-peer communication

# module load python cuda cudnn gcc hdf5
# Activate the virtual environment with all the dependencies
export MODULEPATH=/mnt/home/gkrawezik/modules/rocky8:$MODULEPATH
module load cuda/12.4 cudnn/9.1.0.70-cuda12 nccl/2.21.5-1+cuda12.4
source /mnt/home/mmccabe/venvs/mamba_well/bin/activate


# Launch the training script

srun python -u `which torchrun` \
	--nnodes=$SLURM_JOB_NUM_NODES \
	--nproc_per_node=$SLURM_GPUS_PER_NODE \
	--rdzv_id=$SLURM_JOB_ID \
		--rdzv_backend=c10d \
		--rdzv_endpoint=$SLURMD_NODENAME:29500 \
		train.py distribution=hsdp name=merge_into_main_well trainer.grad_acc_steps=1 server=gpuxl optimizer=adam optimizer.lr=3.5e-4 logger.wandb_project_name="MPPX_Tuning_1B_debug" \
			trainer.enable_amp=False model.gradient_checkpointing_freq=2 trainer.log_interval=2 trainer.clip_gradient=10 data.module_parameters.batch_size=2 data.module_parameters.n_steps_input=6 data.module_parameters.n_steps_output=1 model/processor/space_mixing=full_spatial_attention  \
			model.projection_dim=32 model.intermediate_dim=192 model.hidden_dim=832 model.groups=8 model.processor_blocks=12 model.drop_path=0. \
			model/processor/space_mixing=full_spatial_attention model.processor.space_mixing.num_heads=8 model.processor.time_mixing.num_heads=8 \
			model.causal_in_time=True model.jitter_patches=True data.module_parameters.max_samples=2000 \
			lr_scheduler=inv_sqrt_w_sqrt_ramps data.module_parameters.max_dt_stride=5 \
			trainer.prediction_type="delta" data=light_data trainer.max_epoch=200 data_workers=8 trainer.val_frequency=1 trainer.rollout_val_frequency=1 auto_resume=False
