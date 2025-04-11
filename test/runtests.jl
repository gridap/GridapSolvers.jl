using GridapSolvers
using Test

@testset "MultilevelTools" begin include("MultilevelTools/seq/runtests.jl") end
@testset "LinearSolvers" begin include("LinearSolvers/seq/runtests.jl") end
@testset "NonlinearSolvers" begin include("NonlinearSolvers/seq/runtests.jl") end
@testset "BlockSolvers" begin include("BlockSolvers/seq/runtests.jl") end
@testset "Applications" begin include("Applications/seq/runtests.jl") end
