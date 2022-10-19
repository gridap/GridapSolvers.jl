module GridapSolvers

  using MPI
  using LinearAlgebra
  using Gridap
  using Gridap.Helpers
  using PartitionedArrays
  using GridapDistributed
  using GridapP4est



  export ModelHierarchy

  include("PartitionedArraysExtensions.jl")
  include("ModelHierarchies.jl")

end
