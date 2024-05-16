module KrylovTestsSequential
using PartitionedArrays
include("../KrylovTests.jl")

with_debug() do distribute
  KrylovTests.main(distribute,(1,1))   # 2D - serial
  KrylovTests.main(distribute,(2,2))   # 2D
  KrylovTests.main(distribute,(1,1,1)) # 3D - serial
  KrylovTests.main(distribute,(2,2,1)) # 3D
end

end