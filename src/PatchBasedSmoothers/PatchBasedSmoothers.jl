module PatchBasedSmoothers

using FillArrays, BlockArrays
using LinearAlgebra
using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.Arrays, Gridap.Fields
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs

using PartitionedArrays
using GridapDistributed

using GridapSolvers.MultilevelTools

export PatchDecomposition, Closure
export PatchFESpace
export PatchBasedLinearSolver, VankaSolver

export PatchProlongationOperator, PatchRestrictionOperator
export setup_patch_prolongation_operators, setup_patch_restriction_operators

# Geometry
include("seq/PatchDecompositions.jl")
include("mpi/PatchDecompositions.jl")
include("seq/PatchTriangulations.jl")
include("seq/PatchClosures.jl")
include("seq/CoarsePatchDecompositions.jl")

# FESpaces
include("seq/PatchFESpaces.jl")
include("mpi/PatchFESpaces.jl")
include("seq/ZeroMeanPatchFESpaces.jl")
include("seq/PatchMultiFieldFESpaces.jl")

# Solvers
include("seq/PatchBasedLinearSolvers.jl")
include("seq/PatchTransferOperators.jl")
include("seq/VankaSolvers.jl")

end