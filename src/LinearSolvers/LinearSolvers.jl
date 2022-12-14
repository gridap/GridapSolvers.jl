module LinearSolvers

using LinearAlgebra
using Gridap
using Gridap.Algebra
using PartitionedArrays

using GridapSolvers.MultilevelTools

import LinearAlgebra: mul!, ldiv!

export JacobiLinearSolver
export RichardsonSmoother

include("JacobiLinearSolvers.jl")
include("RichardsonSmoothers.jl")
#include("GMGLinearSolvers.jl")

end