module BlockDiagonalSmoothersTestsMPI
using PartitionedArrays, MPI
include("../BlockDiagonalSmoothersTests.jl")

with_mpi() do distribute
  BlockDiagonalSmoothersTests.main(distribute,(2,2),false)
  BlockDiagonalSmoothersTests.main(distribute,(2,2),true)
end

end