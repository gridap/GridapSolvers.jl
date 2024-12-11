module StaggeredFEOperatorsMPITests
using MPI, PartitionedArrays
include("../StaggeredFEOperatorsTests.jl")

with_mpi() do distribute
  StaggeredFEOperatorsTests.main(distribute,(2,2))
end

end