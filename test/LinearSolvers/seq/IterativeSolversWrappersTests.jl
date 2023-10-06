module IterativeSolversWrappersTestsSequential
using PartitionedArrays
include("../IterativeSolversWrappersTests.jl")

with_debug() do distribute
  IterativeSolversWrappersTests.main(distribute,(1,1))   # 2D - serial
  IterativeSolversWrappersTests.main(distribute,(2,2))   # 2D
  IterativeSolversWrappersTests.main(distribute,(1,1,1)) # 3D - serial
  IterativeSolversWrappersTests.main(distribute,(2,2,1)) # 3D
end

end