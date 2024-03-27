module StokesApplicationMPI
using MPI, PartitionedArrays
include("../Stokes.jl")

with_mpi() do distribute
  StokesApplication.main(distribute,(2,2),(8,8))
  StokesApplication.main(distribute,(2,2,1),(4,4,4))
end

end