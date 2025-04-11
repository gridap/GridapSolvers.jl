module IterativeSolversWrappersTestsSequential
using PartitionedArrays
include("../IterativeSolversExtTests.jl")

with_debug() do distribute
  IterativeSolversExtTests.main(distribute,(1,1))   # 2D - serial
  IterativeSolversExtTests.main(distribute,(2,2))   # 2D
  IterativeSolversExtTests.main(distribute,(1,1,1)) # 3D - serial
  IterativeSolversExtTests.main(distribute,(2,2,1)) # 3D
end

end