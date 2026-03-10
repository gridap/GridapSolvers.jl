module SolverInterfaces

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using BlockArrays

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Arrays

using PartitionedArrays
using GridapDistributed
using GridapDistributed: BlockPRange

using AbstractTrees
using Printf

include("GridapExtras.jl")
include("PAExtras.jl")

include("SolverTolerances.jl")
include("ConvergenceLogs.jl")
include("SolverInfos.jl")
include("NullSpaces.jl")

export SolverVerboseLevel, SolverConvergenceFlag
export SolverTolerances, get_solver_tolerances, set_solver_tolerances!
export ConvergenceLog, init!, update!, finalize!, reset!, print_message, set_depth!

export SolverInfo

export NullSpace

end