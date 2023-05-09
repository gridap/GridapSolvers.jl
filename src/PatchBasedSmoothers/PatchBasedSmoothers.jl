module PatchBasedSmoothers

using FillArrays
using LinearAlgebra
using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Geometry
using Gridap.FESpaces

using PartitionedArrays
using GridapDistributed

using GridapSolvers.MultilevelTools

export PatchDecomposition
export PatchFESpace
export PatchBasedLinearSolver

include("seq/PatchDecompositions.jl")
include("seq/PatchTriangulations.jl")
include("seq/PatchFESpaces.jl")
include("seq/PatchBasedLinearSolvers.jl")

include("mpi/PatchDecompositions.jl")
include("mpi/PatchFESpaces.jl")

end