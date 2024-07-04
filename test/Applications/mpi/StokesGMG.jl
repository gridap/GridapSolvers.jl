module StokesGMGApplicationMPI
using MPI, PartitionedArrays
include("../StokesGMG.jl")

with_mpi() do distribute
  StokesGMGApplication.main(distribute,4,(8,8),[(2,2),(1,1)])
  StokesGMGApplication.main(distribute,4,(8,8),[(2,2),(2,1)])
  StokesGMGApplication.main(distribute,4,(8,8),[(2,2),(2,2)])
  StokesGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(1,1,1)])
  StokesGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(2,1,1)])
  StokesGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(2,2,1)])
end

end