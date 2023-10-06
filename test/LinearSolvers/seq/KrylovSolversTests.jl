module KrylovSolversTestsSequential
using PartitionedArrays
include("../KrylovSolversTests.jl")

with_debug() do distribute
  KrylovSolversTests.main(distribute,(1,1))   # 2D - serial
  KrylovSolversTests.main(distribute,(2,2))   # 2D
  KrylovSolversTests.main(distribute,(1,1,1)) # 3D - serial
  KrylovSolversTests.main(distribute,(2,2,1)) # 3D
end

end