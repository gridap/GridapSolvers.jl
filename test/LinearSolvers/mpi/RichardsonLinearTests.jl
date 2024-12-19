module RichardsonLinearTestsMPI
using MPI, PartitionedArrays
include("../RichardsonLinearTests.jl")

with_mpi() do distribute 
  RichardsonLinearTests.main(distribute,(2,2))   # 2D
  RichardsonLinearTests.main(distribute,(2,2,1)) # 3D
end

end