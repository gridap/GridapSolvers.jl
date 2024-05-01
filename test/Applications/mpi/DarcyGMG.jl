module StokesGMGApplicationMPI
using MPI, PartitionedArrays
include("../DarcyGMG.jl")

with_mpi() do distribute
  DarcyGMGApplication.main(distribute,4,(8,8),[4,1])
  DarcyGMGApplication.main(distribute,4,(8,8),[4,2])
  DarcyGMGApplication.main(distribute,4,(8,8),[4,4])
  DarcyGMGApplication.main(distribute,4,(4,4,4),[4,1])
  DarcyGMGApplication.main(distribute,4,(4,4,4),[4,4])
end

end