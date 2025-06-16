using GridapSolvers
using Test

TESTCASE = get(ENV, "TESTCASE", "seq")

# Sequential tests

if TESTCASE ∈ ("all", "seq", "seq-multilevel")
  include("MultilevelTools/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-linear")
  include("LinearSolvers/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-nonlinear")
  include("NonlinearSolvers/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-block")
  include("BlockSolvers/seq/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-applications")
  include("Applications/seq/runtests.jl")
end

# MPI tests

if TESTCASE ∈ ("all", "mpi", "mpi-multilevel")
  include("MultilevelTools/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-linear")
  include("LinearSolvers/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-nonlinear")
  include("NonlinearSolvers/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-block")
  include("BlockSolvers/mpi/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-applications")
  include("Applications/mpi/runtests.jl")
end

# Extensions

if TESTCASE ∈ ("all", "extlibs")
  include("ExtLibraries/runtests.jl")
end
