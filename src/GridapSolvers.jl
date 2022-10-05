module GridapSolvers

  using Gridap
  using Gridap.Helpers
  using PartitionedArrays
  using GridapDistributed

  export ModelHierarchy

  include("ModelHierarchies.jl")

end
