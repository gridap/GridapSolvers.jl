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

using Gridap.Geometry: PatchTopology, get_patch_cells, get_patch_faces
using Gridap.FESpaces: PatchAssembler
using GridapDistributed: DistributedFESpace, DistributedPatchTopology, DistributedPatchAssembler

export PatchSolver, VankaSolver

export PatchProlongationOperator, PatchRestrictionOperator
export setup_patch_prolongation_operators, setup_patch_restriction_operators

# Solvers
include("PatchTransferOperators.jl")
include("VankaSolvers.jl")
include("PatchSolvers.jl")

end