module PatchBasedSmoothers

using FillArrays, BlockArrays
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

end