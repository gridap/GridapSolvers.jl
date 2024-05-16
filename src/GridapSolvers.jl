module GridapSolvers

  include("SolverInterfaces/SolverInterfaces.jl")
  include("MultilevelTools/MultilevelTools.jl")
  include("BlockSolvers/BlockSolvers.jl")
  include("PatchBasedSmoothers/PatchBasedSmoothers.jl")
  include("LinearSolvers/LinearSolvers.jl")
  include("NonlinearSolvers/NonlinearSolvers.jl")

  using GridapSolvers.SolverInterfaces
  using GridapSolvers.MultilevelTools
  using GridapSolvers.BlockSolvers
  using GridapSolvers.LinearSolvers
  using GridapSolvers.PatchBasedSmoothers
  using GridapSolvers.NonlinearSolvers

  # MultilevelTools
  export get_parts, generate_level_parts, generate_subparts

  export ModelHierarchy, CartesianModelHierarchy
  export num_levels, get_level, get_level_parts
  export get_model, get_model_before_redist

  export FESpaceHierarchy
  export get_fe_space, get_fe_space_before_redist
  export compute_hierarchy_matrices

  export DistributedGridTransferOperator
  export RestrictionOperator, ProlongationOperator
  export setup_transfer_operators

  # BlockSolvers
  export BlockDiagonalSolver

  # LinearSolvers
  export JacobiLinearSolver
  export RichardsonSmoother
  export SymGaussSeidelSmoother
  export GMGLinearSolver
  export BlockDiagonalSmoother

  export ConjugateGradientSolver
  export IS_GMRESSolver
  export IS_MINRESSolver
  export IS_SSORSolver

  export CGSolver
  export MINRESSolver
  export GMRESSolver
  export FGMRESSolver

  # PatchBasedSmoothers
  export PatchDecomposition
  export PatchFESpace
  export PatchBasedLinearSolver

end
