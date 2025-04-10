module ModelHierarchiesTestsSeq
using PartitionedArrays
include("../ModelHierarchiesTests.jl")

with_debug() do distribute
  ModelHierarchiesTests.main(distribute,4,[(2,2),(2,2),(2,1)])
end

end