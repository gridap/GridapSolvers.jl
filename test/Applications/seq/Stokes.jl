module StokesApplicationSequential
using PartitionedArrays
include("../Stokes.jl")

with_debug() do distribute
  StokesApplication.main(distribute,(1,1),(8,8))
  StokesApplication.main(distribute,(2,2),(8,8))
  StokesApplication.main(distribute,(2,2,1),(4,4,4))
end

end