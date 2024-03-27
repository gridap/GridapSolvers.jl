module SmoothersTestsMPI
using MPI, PartitionedArrays
include("../SmoothersTests.jl")

with_mpi() do distribute
  SmoothersTests.main(distribute,(2,2))   # 2D
  SmoothersTests.main(distribute,(2,2,1)) # 3D
end

end