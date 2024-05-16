module BlockDiagonalSolversMPITests
using MPI, PartitionedArrays
include("../BlockDiagonalSolversTests.jl")

with_mpi() do distribute
  BlockDiagonalSolversTests.main(distribute,(2,2))
end

end