module PatchBasedSmoothers

using FillArrays, BlockArrays
using LinearAlgebra
using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.Arrays, Gridap.Fields
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs

using PartitionedArrays
using GridapDistributed

using GridapSolvers.MultilevelTools
using GridapSolvers.MultilevelTools: get_cell_conformity

using Gridap.Geometry: PatchTopology
using Gridap.FESpaces: PatchAssembler
using GridapDistributed: DistributedFESpace, DistributedPatchTopology, DistributedPatchAssembler

export PatchDecomposition, Closure
export PatchFESpace
export PatchBasedLinearSolver, VankaSolver

export PatchProlongationOperator, PatchRestrictionOperator
export setup_patch_prolongation_operators, setup_patch_restriction_operators

# Geometry
include("PatchDecompositions.jl")
include("DistributedPatchDecompositions.jl")
include("PatchTriangulations.jl")
include("PatchClosures.jl")

# FESpaces
include("PatchFESpaces.jl")
include("DistributedPatchFESpaces.jl")
include("ZeroMeanPatchFESpaces.jl")
include("PatchMultiFieldFESpaces.jl")

# Solvers
include("PatchBasedLinearSolvers.jl")
include("PatchTransferOperators.jl")
include("VankaSolvers.jl")
include("PatchSolvers.jl")

end