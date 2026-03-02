module NavierStokesGMGApplicationMPI
using MPI, PartitionedArrays
include("../NavierStokesGMG.jl")

with_mpi() do distribute
  NavierStokesGMGApplication.main(distribute,4,(8,8),[(2,2),(1,1)])
  NavierStokesGMGApplication.main(distribute,4,(8,8),[(2,2),(2,1)])
  NavierStokesGMGApplication.main(distribute,4,(8,8),[(2,2),(2,2)])
  NavierStokesGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(1,1,1)])
  NavierStokesGMGApplication.main(distribute,4,(4,4,4),[(2,2,1),(2,2,1)])
end

end