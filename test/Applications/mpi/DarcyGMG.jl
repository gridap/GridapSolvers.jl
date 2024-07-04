module StokesGMGApplicationMPI
using MPI, PartitionedArrays
include("../DarcyGMG.jl")

with_mpi() do distribute
  DarcyGMGApplication.main(distribute,4,(8,8),[(2,2),(1,1)])
  DarcyGMGApplication.main(distribute,4,(8,8),[(2,2),(2,1)])
  DarcyGMGApplication.main(distribute,4,(8,8),[(2,2),(2,2)])
  DarcyGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(1,1,1)])
  DarcyGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(2,2,1)])
end

end