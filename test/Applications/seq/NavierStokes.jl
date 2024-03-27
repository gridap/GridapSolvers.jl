module NavierStokesApplicationSequential
using PartitionedArrays
include("../NavierStokes.jl")

with_debug() do distribute
  NavierStokesApplication.main(distribute,(1,1),(8,8))
  NavierStokesApplication.main(distribute,(2,2),(8,8))
  NavierStokesApplication.main(distribute,(2,2,1),(4,4,4))
end

end