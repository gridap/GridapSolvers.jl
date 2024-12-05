module SmoothersTestsMPI
using MPI, PartitionedArrays
include("../SchwarzSolversTests.jl")

# Deactivated for now. We require sub-assembled matrices for this
with_mpi() do distribute
  # SchwarzSolversTests.main(distribute,(2,2))   # 2D
  # SchwarzSolversTests.main(distribute,(2,2,1)) # 3D
end

end