using GridapSolvers
using Test

include("MultilevelTools/mpi/runtests.jl")
include("LinearSolvers/mpi/runtests.jl")
include("BlockSolvers/mpi/runtests.jl")
include("Applications/mpi/runtests.jl")
