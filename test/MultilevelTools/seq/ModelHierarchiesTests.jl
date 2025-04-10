module ModelHierarchiesTestsSeq
using PartitionedArrays
include("../ModelHierarchiesTests.jl")

with_debug() do distribute
  ModelHierarchiesTests.main(distribute,4,[4,4,2,2])
end

end