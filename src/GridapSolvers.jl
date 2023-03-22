module GridapSolvers

  include("MultilevelTools/MultilevelTools.jl")
  include("LinearSolvers/LinearSolvers.jl")
  include("PatchBasedSmoothers/PatchBasedSmoothers.jl")

  using GridapSolvers.MultilevelTools
  using GridapSolvers.LinearSolvers
  using GridapSolvers.PatchBasedSmoothers

  # MultilevelTools
  export get_parts, generate_level_parts, generate_subparts

  export ModelHierarchy
  export num_levels, get_level, get_level_parts
  export get_model, get_model_before_redist

  export FESpaceHierarchy
  export get_fe_space, get_fe_space_before_redist
  export compute_hierarchy_matrices

  export DistributedGridTransferOperator
  export RestrictionOperator, ProlongationOperator
  export setup_transfer_operators

  # LinearSolvers
  export JacobiLinearSolver
  export RichardsonSmoother
  export GMGLinearSolver
  export BlockDiagonalSmoother

  # PatchBasedSmoothers
  export PatchDecomposition
  export PatchFESpace
  export PatchBasedLinearSolver

end
