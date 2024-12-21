module RichardsonLinearTestsSequential
using PartitionedArrays
include("../RichardsonLinearTests.jl")

with_debug() do distribute
  RichardsonLinearTests.main(distribute,(1,1))   # 2D - serial
  RichardsonLinearTests.main(distribute,(2,2))   # 2D
  RichardsonLinearTests.main(distribute,(1,1,1)) # 3D - serial
  RichardsonLinearTests.main(distribute,(2,2,1)) # 3D 
end

end