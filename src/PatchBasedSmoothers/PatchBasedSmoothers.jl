module PatchBasedSmoothers

using FillArrays, BlockArrays
using LinearAlgebra
using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.Arrays
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs

using PartitionedArrays
using GridapDistributed

using GridapSolvers.MultilevelTools

export PatchDecomposition
export PatchFESpace
export PatchBasedLinearSolver

export PatchProlongationOperator, PatchRestrictionOperator
export setup_patch_prolongation_operators, setup_patch_restriction_operators

# Geometry
include("seq/PatchDecompositions.jl")
include("mpi/PatchDecompositions.jl")
include("seq/PatchTriangulations.jl")

# FESpaces
include("seq/PatchFESpaces.jl")
include("mpi/PatchFESpaces.jl")
include("seq/PatchMultiFieldFESpaces.jl")

# Solvers
include("seq/PatchBasedLinearSolvers.jl")
include("seq/PatchTransferOperators.jl")

end