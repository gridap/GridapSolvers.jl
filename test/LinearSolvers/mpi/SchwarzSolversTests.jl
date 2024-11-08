module SmoothersTestsMPI
using MPI, PartitionedArrays
include("../SchwarzSolversTests.jl")

with_mpi() do distribute
  SchwarzSolversTests.main(distribute,(2,2))   # 2D
  SchwarzSolversTests.main(distribute,(2,2,1)) # 3D
end

end