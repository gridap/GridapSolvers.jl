module RefinementToolsTestsMPI
using MPI, PartitionedArrays
include("../RefinementToolsTests.jl")

with_mpi() do distribute
  RefinementToolsTests.main(distribute,4,2,[(2,2),(2,1),(2,1)]) # 2D
  #RefinementToolsTests.main(distribute,4,3,[4,2,2]) # 3D
end

end