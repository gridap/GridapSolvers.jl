module DistributedGridTransferOperatorsTestsMPI
using MPI, PartitionedArrays
include("../DistributedGridTransferOperatorsTests.jl")

with_mpi() do distribute
  DistributedGridTransferOperatorsTests.main(distribute,4,2,[4,2,2]) # 2D
  #DistributedGridTransferOperatorsTests.main(distribute,4,3,[4,2,2]) # 3D
end

end