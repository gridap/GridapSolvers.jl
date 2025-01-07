module NullspaceTestsSequential
using PartitionedArrays
include("../NullspaceTests.jl")

@testset "NullspaceTests" begin
  NullspaceTests.main_interfaces()
  NullspaceTests.main()
end

end