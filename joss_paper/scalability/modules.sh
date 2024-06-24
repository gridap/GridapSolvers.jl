module purge
module load pbs

module load intel-compiler-llvm/2023.2.0
module load intel-mpi/2021.10.0
module load intel-mkl/2023.2.0

export P4EST_VERSION='2.8.5'
export PETSC_VERSION='3.19.5'
export PROJECT="np01"

SCRATCH="/scratch/$PROJECT/$USER"
export JULIA_DEPOT_PATH="$SCRATCH/.julia"
export MPI_VERSION="intel-$INTEL_MPI_VERSION"
export JULIA_MPI_PATH=$INTEL_MPI_ROOT
export JULIA_PETSC_LIBRARY="$HOME/bin/petsc/$PETSC_VERSION-$MPI_VERSION/lib/libpetsc"
export P4EST_ROOT_DIR="$HOME/bin/p4est/$P4EST_VERSION-$MPI_VERSION"

export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export HCOLL_ML_DISABLE_SCATTERV=1
export HCOLL_ML_DISABLE_BCAST=1
export ZES_ENABLE_SYSMAN=1