module BlockSolvers
  using LinearAlgebra
  using SparseArrays
  using SparseMatricesCSR
  using BlockArrays

  using Gridap
  using Gridap.Helpers, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.FESpaces, Gridap.MultiField
  using PartitionedArrays
  using GridapDistributed

  using GridapSolvers.MultilevelTools
  using GridapSolvers.SolverInterfaces

  include("BlockFEOperators.jl")

  include("BlockSolverInterfaces.jl")
  include("BlockDiagonalSolvers.jl")
  include("BlockTriangularSolvers.jl")

  export BlockFEOperator

  export MatrixBlock, LinearSystemBlock, NonlinearSystemBlock, BiformBlock, TriformBlock

  export BlockDiagonalSolver
  export BlockTriangularSolver
end
