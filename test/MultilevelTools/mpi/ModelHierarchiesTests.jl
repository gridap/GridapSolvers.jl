module ModelHierarchiesTestsMPI
using MPI, PartitionedArrays
include("../ModelHierarchiesTests.jl")

with_mpi() do distribute
  ModelHierarchiesTests.main(distribute,4,[(2,2),(2,2),(2,1),(1,1)])
end

end