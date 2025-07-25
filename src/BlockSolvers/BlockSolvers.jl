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

using Gridap.MultiField: BlockSparseMatrixAssembler
using Gridap.MultiField: split_fespace, combine_fespaces

using GridapDistributed: to_parray_of_arrays
using GridapDistributed: DistributedFESpace, DistributedSingleFieldFESpace
using GridapDistributed: DistributedMultiFieldFESpace, DistributedMultiFieldFEFunction

const MultiFieldFESpaceTypes = Union{<:MultiFieldFESpace,<:DistributedMultiFieldFESpace}
const BlockFESpaceTypes{NB,SB,P} = Union{<:MultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}},<:DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}}

include("BlockSolverInterfaces.jl")
include("BlockDiagonalSolvers.jl")
include("BlockTriangularSolvers.jl")

include("BlockFEOperators.jl")
include("StaggeredFEOperators.jl")

export MatrixBlock, LinearSystemBlock, NonlinearSystemBlock, BiformBlock, TriformBlock

export BlockDiagonalSolver
export BlockTriangularSolver

export BlockFEOperator
export StaggeredFEOperator, StaggeredAffineFEOperator, StaggeredNonlinearFEOperator, StaggeredFESolver

end
