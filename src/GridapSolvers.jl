module GridapSolvers

  using MPI
  using LinearAlgebra
  using Gridap
  using Gridap.Helpers
  using Gridap.Geometry
  using Gridap.FESpaces
  using PartitionedArrays
  using GridapDistributed
  using GridapP4est

  import GridapDistributed: local_views


  export ModelHierarchy
  export num_levels, get_level, get_model, get_model_before_redist, has_refinement, has_redistribution

  export FESpaceHierarchy
  export get_space, get_space_before_redist

  export InterGridTransferOperator
  export RestrictionOperator, ProlongationOperator
  export setup_transfer_operators

  include("PartitionedArraysExtensions.jl")
  include("ModelHierarchies.jl")
  include("FESpaceHierarchies.jl")
  include("RedistributeTools.jl")
  include("InterGridTransferOperators.jl")


end
