module GridapSolvers

  using MPI
  using Gridap
  using Gridap.Helpers
  using PartitionedArrays
  using GridapDistributed
  using GridapP4est



  export ModelHierarchy
  export model_hierarchy_free!

  include("PartitionedArraysExtensions.jl")
  include("ModelHierarchies.jl")

end
