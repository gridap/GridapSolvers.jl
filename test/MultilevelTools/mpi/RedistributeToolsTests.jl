module RedistributeToolsTestsMPI
using MPI, PartitionedArrays
include("../RedistributeToolsTests.jl")

with_mpi() do distribute
  RedistributeToolsTests.main(distribute,4,2,[4,2]) # 2D
  #RedistributeToolsTests.main(distribute,4,3,[4,2]) # 3D
end

end