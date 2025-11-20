# module load modules/2.3-20240529  openmpi/4.1.6 hdf5/mpi-1.14.3
module load modules/2.3-20240529  openmpi/4.1.6 hdf5/mpi-1.8.23 python-mpi/3.11.7 cuda/12.3.2 cudnn/8.9.7.29-12 
source ~/venvs/mpi_well
# module load cuda cudnn gcc hdf5
export HYDRA_FULL_ERROR=1
export HDF5_USE_FILE_LOCKING=FALSE
