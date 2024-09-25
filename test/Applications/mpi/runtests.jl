using Test
using MPI
using GridapSolvers

function run_tests(testdir)
  istest(f) = endswith(f, ".jl") && !(f=="runtests.jl")
  testfiles = sort(filter(istest, readdir(testdir)))
  @time @testset "$f" for f in testfiles
    MPI.mpiexec() do cmd
      np = 4
      cmd = `$cmd -n $(np) --oversubscribe $(Base.julia_cmd()) --project=. $(joinpath(testdir, f))`
      @show cmd
      run(cmd)
      @test true
    end
  end
end

# MPI tests
run_tests(@__DIR__)