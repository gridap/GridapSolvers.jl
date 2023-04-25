module LinearSolvers

using Printf
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using BlockArrays
using IterativeSolvers

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using PartitionedArrays
using GridapPETSc

using GridapSolvers.MultilevelTools

import LinearAlgebra: mul!, ldiv!

export JacobiLinearSolver
export RichardsonSmoother
export SymGaussSeidelSmoother
export GMGLinearSolver
export BlockDiagonalSmoother

# Wrappers for IterativeSolvers.jl
export IS_ConjugateGradientSolver
export IS_GMRESSolver
export IS_MINRESSolver
export IS_SSORSolver

export GMRESSolver

include("Helpers.jl")
include("JacobiLinearSolvers.jl")
include("RichardsonSmoothers.jl")
include("SymGaussSeidelSmoothers.jl")
include("GMGLinearSolvers.jl")
include("BlockDiagonalSmoothers.jl")
include("IterativeLinearSolvers.jl")
include("GMRESSolvers.jl")

end