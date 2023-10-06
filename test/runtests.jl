using GridapSolvers
using Test

@testset "Sequential tests" begin
  include("MultilevelTools/seq/runtests.jl")
  include("LinearSolvers/seq/runtests.jl")
end

@testset "MPI tests" begin
  include("MultilevelTools/mpi/runtests.jl")
  include("LinearSolvers/mpi/runtests.jl")
end
