module BlockDiagonalSmoothersTestsSeq
using PartitionedArrays
include("../BlockDiagonalSmoothersTests.jl")

with_debug() do distribute
  BlockDiagonalSmoothersTests.main(distribute,(1,1),false)
  BlockDiagonalSmoothersTests.main(distribute,(1,1),true)
  BlockDiagonalSmoothersTests.main(distribute,(2,2),false)
  BlockDiagonalSmoothersTests.main(distribute,(2,2),true)
end

end