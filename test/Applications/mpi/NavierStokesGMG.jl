module NavierStokesGMGApplicationMPI
using MPI, PartitionedArrays
include("../NavierStokesGMG.jl")

with_mpi() do distribute
  NavierStokesGMGApplication.main(distribute,4,(8,8))
  #NavierStokesGMGApplication.main(distribute,4,(4,4,4))
end

end