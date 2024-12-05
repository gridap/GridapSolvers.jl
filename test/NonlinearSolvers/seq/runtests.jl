
using Test
using PartitionedArrays

include("../NonlinearSolversTests.jl")

@testset "NonlinearSolvers - Serial" begin
  @testset "NLSolvers" begin
    NonlinearSolversTests.main(:nlsolvers_newton)
    NonlinearSolversTests.main(:nlsolvers_trust_region)
    NonlinearSolversTests.main(:nlsolvers_anderson)
  end
  @testset "Newton" begin
    NonlinearSolversTests.main(:newton)
    NonlinearSolversTests.main(:newton_continuation)
  end
end

@testset "NonlinearSolvers - Sequential" begin
  @testset "NLSolvers" begin
    NonlinearSolversTests.main(DebugArray,4,:nlsolvers_newton)
    NonlinearSolversTests.main(DebugArray,4,:nlsolvers_trust_region)
    NonlinearSolversTests.main(DebugArray,4,:nlsolvers_anderson)
  end
  @testset "Newton" begin
    NonlinearSolversTests.main(DebugArray,4,:newton)
    NonlinearSolversTests.main(DebugArray,4,:newton_continuation)
  end
end
