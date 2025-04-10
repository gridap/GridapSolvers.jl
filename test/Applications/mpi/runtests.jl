using Test
using MPI
using GridapSolvers

function run_tests(testdir,procs=4)
  istest(f) = endswith(f, ".jl") && !(f=="runtests.jl")
  testfiles = sort(filter(istest, readdir(testdir)))
  @time @testset "$f" for f in testfiles
    MPI.mpiexec() do cmd
      if MPI.MPI_LIBRARY == "OpenMPI" || (isdefined(MPI, :OpenMPI) && MPI.MPI_LIBRARY == MPI.OpenMPI)
        run(`$cmd -n $procs --oversubscribe $(Base.julia_cmd()) --project=. $(f)`)
      else
        run(`$cmd -n $procs $(Base.julia_cmd()) --project=. $(f)`)
      end
      # This line will be reached if and only if the command launched by `run` runs without errors.
      # Then, if we arrive here, the test has succeeded.
      @test true
    end
  end
end

# MPI tests
run_tests(@__DIR__)