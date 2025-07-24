using PartitionedArrays

include("../Applications/Elasticity.jl")

with_mpi() do distribute
  PETScElasticitySolverTests.main(distribute,(2,2))
end

include("drivers/HPDDMTests.jl")

with_mpi() do distribute
  HPDDMTests.main(distribute,(2,2))
end
