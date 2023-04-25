using GridapSolvers
using Test
using ArgParse
using MPI

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--image-file", "-i"
        help = "Path to the image file that one can use in order to accelerate MPI tests"
        arg_type = String
        default="GridapDistributed.so"
    end
    return parse_args(s)
end

"""
  run_tests(testdir)
"""
function run_tests(testdir)
    parsed_args = parse_commandline()
    image_file_path=parsed_args["image-file"]
    image_file_exists=isfile(image_file_path)

    nprocs_str = get(ENV, "JULIA_GRIDAP_SOLVERS_TEST_NPROCS","")
    nprocs = nprocs_str == "" ? clamp(Sys.CPU_THREADS, 2, 4) : parse(Int, nprocs_str)
    istest(f) = endswith(f, ".jl") && !(f=="runtests.jl")
    testfiles = sort(filter(istest, readdir(testdir)))
    @time @testset "$f" for f in testfiles
      MPI.mpiexec() do cmd
        if f in ["DistributedGridTransferOperatorsTests.jl",
                 "RedistributeToolsTests.jl",
                 "RefinementToolsTests.jl",
                 "RichardsonSmoothersTests.jl",
                 "GMGLinearSolversPoissonTests.jl",
                 "GMGLinearSolversLaplacianTests.jl",
                 "GMGLinearSolversVectorLaplacianTests.jl",
                 "GMGLinearSolversHDivRTTests.jl",
                 "MUMPSSolversTests.jl",
                 "GMGLinearSolversMUMPSTests.jl",
                 "RestrictDofsTests.jl",
                 "PRefinementGMGLinearSolversPoissonTests.jl"]
          np = 4
          #extra_args = "-s 2 2 -r 2"
          extra_args = ""
        elseif f in ["ModelHierarchiesTests.jl"]
          np = 6
          extra_args = ""
        else
          np = 4 # nprocs
          extra_args = ""
        end
        if ! image_file_exists
          cmd = `$cmd -n $(np) --allow-run-as-root --oversubscribe $(Base.julia_cmd()) --project=. $(joinpath(testdir, f)) $(split(extra_args))`
        else
          cmd = `$cmd -n $(np) --allow-run-as-root --oversubscribe $(Base.julia_cmd()) -J$(image_file_path) --project=. $(joinpath(testdir, f)) $(split(extra_args))`
        end
        @show cmd
        run(cmd)
        @test true
      end
    end
end

# MPI tests
run_tests(joinpath(@__DIR__, "mpi"))

# Sequential tests
@time @testset "BlockDiagonalSmoothersPETScTests" begin include("seq/BlockDiagonalSmoothersPETScTests.jl") end
@time @testset "BlockDiagonalSmoothersTests" begin include("seq/BlockDiagonalSmoothersTests.jl") end
@time @testset "DistributedPatchFESpacesTests" begin include("seq/DistributedPatchFESpacesTests.jl") end
@time @testset "GMRESSolversTests" begin include("seq/GMRESSolversTests.jl") end
@time @testset "IterativeSolversTests" begin include("seq/IterativeSolversTests.jl") end
@time @testset "PatchLinearSolverTests" begin include("seq/PatchLinearSolverTests.jl") end
@time @testset "SymGaussSeidelSmoothersTests" begin include("seq/SymGaussSeidelSmoothersTests.jl") end
