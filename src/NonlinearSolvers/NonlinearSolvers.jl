module NonlinearSolvers
  using LinearAlgebra
  using SparseArrays
  using SparseMatricesCSR
  using BlockArrays
  using NLsolve, LineSearches

  using Gridap
  using Gridap.Helpers, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.FESpaces, Gridap.MultiField
  using PartitionedArrays
  using GridapDistributed

  using GridapSolvers.SolverInterfaces
  using GridapSolvers.MultilevelTools

  include("NewtonRaphsonSolver.jl")
  export NewtonSolver

  include("NLsolve.jl")
  export NLsolveNonlinearSolver

end