module SolverInterfaces

using Gridap
using Gridap.Helpers
using Gridap.Algebra

using AbstractTrees
using Printf

include("GridapExtras.jl")
include("SolverTolerances.jl")
include("ConvergenceLogs.jl")
include("SolverInfos.jl")

export SolverVerboseLevel, SolverConvergenceFlag
export SolverTolerances, get_solver_tolerances, set_solver_tolerances!
export ConvergenceLog, init!, update!, finalize!, reset!

export SolverInfo

end