module GridapSolvers

  using MPI
  using Gridap
  using Gridap.Helpers
  using PartitionedArrays
  using GridapDistributed



  export ModelHierarchy

  include("PartitionedArraysExtensions.jl")
  include("ModelHierarchies.jl")

end
