module StokesGMGApplicationMPI
using MPI, PartitionedArrays
include("../StokesGMG.jl")

with_mpi() do distribute
  StokesGMGApplication.main(distribute,4,(8,8))
  StokesGMGApplication.main(distribute,4,(4,4,4))
end

end