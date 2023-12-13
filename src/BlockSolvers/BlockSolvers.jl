module BlockSolvers
  using LinearAlgebra
  using SparseArrays
  using SparseMatricesCSR
  using BlockArrays
  using IterativeSolvers

  using Gridap
  using Gridap.Helpers, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.FESpaces, Gridap.MultiField
  using PartitionedArrays
  using GridapDistributed

  using GridapSolvers.MultilevelTools
  using GridapSolvers.SolverInterfaces

  include("BlockSolverInterfaces.jl")
  include("BlockDiagonalSolvers.jl")
  include("BlockTriangularSolvers.jl")

  export BlockDiagonalSolver
  export BlockTriangularSolver
end
