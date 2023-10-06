module KrylovSolversTestsMPI
using MPI, PartitionedArrays
include("../KrylovSolversTests.jl")

with_mpi() do distribute 
  KrylovSolversTests.main(distribute,(2,2))   # 2D
  KrylovSolversTests.main(distribute,(2,2,1)) # 3D
end

end