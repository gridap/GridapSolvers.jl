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

export ConvergenceLog, init!, update!, finalize!, reset!

export SolverInfo, SolverTolerances
export SolverVerboseLevel, SolverLogLevel, SolverConvergenceFlag

end