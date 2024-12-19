module LinearSolvers

using Printf
using AbstractTrees
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using BlockArrays
using IterativeSolvers

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.CellData, Gridap.Arrays, Gridap.FESpaces, Gridap.MultiField
using PartitionedArrays
using GridapPETSc

using GridapDistributed
using GridapSolvers.MultilevelTools
using GridapSolvers.SolverInterfaces
using GridapSolvers.PatchBasedSmoothers

export LinearSolverFromSmoother
export JacobiLinearSolver
export RichardsonSmoother
export SymGaussSeidelSmoother
export GMGLinearSolver
export BlockDiagonalSmoother
export SchurComplementSolver
export SchwarzLinearSolver
export RichardsonLinearSolver

export CallbackSolver

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

include("PETSc/PETScUtils.jl")
include("PETSc/PETScCaches.jl")
include("PETSc/ElasticitySolvers.jl")
include("PETSc/HipmairXuSolvers.jl")

include("IdentityLinearSolvers.jl")

include("LinearSolverFromSmoothers.jl")
include("JacobiLinearSolvers.jl")
include("RichardsonSmoothers.jl")
include("SymGaussSeidelSmoothers.jl")
include("RichardsonLinearSolvers.jl")

include("GMGLinearSolvers.jl")
include("IterativeLinearSolvers.jl")
include("SchurComplementSolvers.jl")
include("MatrixSolvers.jl")
include("SchwarzLinearSolvers.jl")

include("CallbackSolver.jl")

end