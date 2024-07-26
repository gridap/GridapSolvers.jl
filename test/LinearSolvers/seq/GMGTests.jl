module GMGTestsSeq
using PartitionedArrays
include("../GMGTests.jl")

with_debug() do distribute
  GMGTests.main(distribute,4,(4,4),[(2,2),(1,1)]) # 2D
  GMGTests.main(distribute,4,(2,2,2),[(2,2,1),(1,1,1)]) # 3D
end

end