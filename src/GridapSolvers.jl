module GridapSolvers

  using MPI
  using LinearAlgebra
  using FillArrays
  using Gridap
  using Gridap.Helpers
  using Gridap.Algebra
  using Gridap.Geometry
  using Gridap.FESpaces
  using Gridap.Adaptivity
  using PartitionedArrays
  using GridapDistributed
  using GridapP4est

  import GridapDistributed: local_views

  export change_parts, void

  export DistributedRefinedDiscreteModel

  export ModelHierarchy
  export num_levels, get_level, get_level_parts
  export get_model, get_model_before_redist, has_refinement, has_redistribution

  export FESpaceHierarchy
  export get_fe_space, get_fe_space_before_redist

  #export DistributedGridTransferOperator
  #export RestrictionOperator, ProlongationOperator
  #export setup_transfer_operators

  include("PartitionedArraysExtensions.jl")
  include("GridapDistributedExtensions.jl")
  include("GridapFixes.jl")
  include("RefinementTools.jl")
  include("RedistributeTools.jl")
  include("ModelHierarchies.jl")
  include("FESpaceHierarchies.jl")
  #include("DistributedGridTransferOperators.jl")


end
