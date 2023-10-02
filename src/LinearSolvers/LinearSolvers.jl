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
using Gridap.FESpaces
using Gridap.MultiField
using PartitionedArrays
using GridapPETSc

using GridapDistributed
using GridapSolvers.MultilevelTools
using GridapSolvers.SolverInterfaces

export JacobiLinearSolver
export RichardsonSmoother
export SymGaussSeidelSmoother
export GMGLinearSolver
export BlockDiagonalSmoother
export SchurComplementSolver

# Wrappers for IterativeSolvers.jl
export IS_ConjugateGradientSolver
export IS_GMRESSolver
export IS_MINRESSolver
export IS_SSORSolver

# Krylov solvers 
export CGSolver
export GMRESSolver
export FGMRESSolver
export MINRESSolver

include("Krylov/KrylovUtils.jl")
include("Krylov/CGSolvers.jl")
include("Krylov/GMRESSolvers.jl")
include("Krylov/FGMRESSolvers.jl")
include("Krylov/MINRESSolvers.jl")

include("IdentityLinearSolvers.jl")
include("JacobiLinearSolvers.jl")
include("RichardsonSmoothers.jl")
include("SymGaussSeidelSmoothers.jl")
include("GMGLinearSolvers.jl")
include("BlockDiagonalSmoothers.jl")
include("IterativeLinearSolvers.jl")
include("SchurComplementSolvers.jl")

end