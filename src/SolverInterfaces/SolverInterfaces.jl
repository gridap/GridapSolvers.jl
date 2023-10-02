module SolverInterfaces

using Gridap
using Gridap.Helpers
using Gridap.Algebra

using AbstractTrees

include("GridapExtras.jl")
include("SolverTolerances.jl")
include("SolverInfos.jl")

export SolverInfo, SolverTolerances
export SolverVerboseLevel, SolverLogLevel, SolverConvergenceFlag

end