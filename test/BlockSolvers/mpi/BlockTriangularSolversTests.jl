module BlockTriangularSolversMPITests
using MPI, PartitionedArrays
include("../BlockTriangularSolversTests.jl")

with_mpi() do distribute
  BlockTriangularSolversTests.main(distribute,(2,2))
end

end