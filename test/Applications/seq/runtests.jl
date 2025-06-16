using Test

@testset "Applications" begin

  @testset "Stokes" begin include("Stokes.jl") end
  @testset "Navier-Stokes" begin include("NavierStokes.jl") end

end
