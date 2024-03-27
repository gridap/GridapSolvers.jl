module KrylovTestsMPI
using MPI, PartitionedArrays
include("../KrylovTests.jl")

with_mpi() do distribute 
  KrylovTests.main(distribute,(2,2))   # 2D
  KrylovTests.main(distribute,(2,2,1)) # 3D
end

end