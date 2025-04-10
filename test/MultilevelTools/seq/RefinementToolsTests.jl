module RefinementToolsTestsSeq
using PartitionedArrays
include("../RefinementToolsTests.jl")

with_debug() do distribute
  RefinementToolsTests.main(distribute,4,2,[4,2,2]) # 2D
  #RefinementToolsTests.main(distribute,4,3,[4,2,2]) # 3D
end

end 