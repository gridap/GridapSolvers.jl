
using Test

include("../NonlinearSolversTests.jl")

@testset "NonlinearSolvers" begin
  @testset "NLSolvers" begin
    NonlinearSolversTests.main(:nlsolvers_newton)
    NonlinearSolversTests.main(:nlsolvers_trust_region)
    NonlinearSolversTests.main(:nlsolvers_anderson)
  end
  @testset "Newton" begin
    NonlinearSolversTests.main(:newton)
  end
end
