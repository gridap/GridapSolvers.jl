using Test

@testset "LinearSolvers" begin

  include("KrylovTests.jl")
  include("IterativeSolversExtTests.jl")
  include("SmoothersTests.jl")
  include("GMGTests.jl")
  include("RichardsonLinearTests.jl")
  include("PardisoExtTests.jl")

end
