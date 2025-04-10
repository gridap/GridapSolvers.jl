module RedistributeToolsTestsSeq
using PartitionedArrays
include("../RedistributeToolsTests.jl")

with_debug() do distribute
  RedistributeToolsTests.main(distribute,4,2,[4,2]) # 2D
  #RedistributeToolsTests.main(distribute,4,3,[4,2]) # 3D
end

end