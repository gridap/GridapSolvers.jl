module CondensedBlockJacobiTestsSequential
using PartitionedArrays
include("../CondensedBlockJacobiTests.jl")

CondensedBlockJacobiTests.main(2)
CondensedBlockJacobiTests.main(3)

with_debug() do distribute
  CondensedBlockJacobiTests.main(distribute, (2,2))
  CondensedBlockJacobiTests.main(distribute, (2,2,1))
end

end
