using PartitionedArrays

include("../Applications/Elasticity.jl")

with_mpi() do distribute
  main(distribute,(2,2))
end
