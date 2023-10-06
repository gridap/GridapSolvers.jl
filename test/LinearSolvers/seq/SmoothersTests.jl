module SmoothersTestsSequential
using PartitionedArrays
include("../SmoothersTests.jl")

with_debug() do distribute
  SmoothersTests.main(distribute,(1,1))   # 2D - serial
  SmoothersTests.main(distribute,(2,2))   # 2D
  SmoothersTests.main(distribute,(1,1,1)) # 3D - serial
  SmoothersTests.main(distribute,(2,2,1)) # 3D
end

end