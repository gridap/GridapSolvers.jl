module BlockDiagonalSolversSequentialTests
using PartitionedArrays
include("../BlockDiagonalSolversTests.jl")

with_debug() do distribute
  BlockDiagonalSolversTests.main(distribute, (1,1))
  BlockDiagonalSolversTests.main(distribute, (2,2))
end

end