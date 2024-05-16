module BlockTriangularSolversSequentialTests

using PartitionedArrays
include("../BlockTriangularSolversTests.jl")

with_debug() do distribute
  BlockTriangularSolversTests.main(distribute, (1,1))
  BlockTriangularSolversTests.main(distribute, (2,2))
end

end