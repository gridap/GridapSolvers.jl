module SchurComplementSolversTestsSequential
using PartitionedArrays
include("../SchurComplementSolversTests.jl")

with_debug() do distribute
  SchurComplementSolversTests.main(distribute,(1,1))
  SchurComplementSolversTests.main(distribute,(2,2))
end

end