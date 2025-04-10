module DistributedGridTransferOperatorsTestsSeq
using PartitionedArrays
include("../DistributedGridTransferOperatorsTests.jl")

with_debug() do distribute
  DistributedGridTransferOperatorsTests.main(distribute,4,2,[4,2,2]) # 2D
  #DistributedGridTransferOperatorsTests.main(distribute,4,3,[4,2,2]) # 3D
end

end