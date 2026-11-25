module CondensedBlockJacobiTestsMPI
using MPI, PartitionedArrays
include("../CondensedBlockJacobiTests.jl")

with_mpi() do distribute
  CondensedBlockJacobiTests.main(distribute, (2,2))
  CondensedBlockJacobiTests.main(distribute, (2,2,1))
end

end
