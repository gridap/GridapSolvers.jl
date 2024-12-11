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

using GridapDistributed: to_parray_of_arrays
using GridapDistributed: DistributedMultiFieldFESpace, DistributedMultiFieldFEFunction

const MultiFieldFESpaceTypes = Union{<:MultiFieldFESpace,<:GridapDistributed.DistributedMultiFieldFESpace}
const BlockFESpaceTypes{NB,SB,P} = Union{<:MultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}},<:GridapDistributed.DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}}

include("BlockSolverInterfaces.jl")
include("BlockDiagonalSolvers.jl")
include("BlockTriangularSolvers.jl")

include("BlockFEOperators.jl")
include("StaggeredFEOperators.jl")

export MatrixBlock, LinearSystemBlock, NonlinearSystemBlock, BiformBlock, TriformBlock

export BlockDiagonalSolver
export BlockTriangularSolver

export BlockFEOperator
export StaggeredFEOperator, StaggeredAffineFEOperator, StaggeredFESolver

end
