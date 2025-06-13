module PardisoExt

using GridapSolvers
using Gridap
using Gridap.Algebra

using Pardiso: MKLPardisoSolver, PardisoSolver
using Pardiso: get_matrix as ps_get_matrix
using Pardiso: set_phase!, set_matrixtype!, set_msglvl!, set_nprocs!
using Pardiso: pardiso, pardisoinit, fix_iparm!
using Pardiso: get_nprocs
using Pardiso: MatrixType, REAL_NONSYM
using Pardiso: MessageLevel, MESSAGE_LEVEL_ON

using Pardiso: ANALYSIS, NUM_FACT, SOLVE_ITERATIVE_REFINE, RELEASE_ALL

# Cannot have name `PardisoSolver` to not conflict with Pardiso.jl, and cannot have name
# `PardisoLinearSolver` to not conflict with the function exported from GridapSolvers.jl.
# Needs to be mutable for the finalizer
mutable struct GridapPardisoSolver{A} <: Algebra.LinearSolver
  ps :: A
  verbose :: Bool
end

"""
    PardisoLinearSolver(; lib = :MKL, mtype = REAL_NONSYM, nthreads = 1, verbose = false)

Interface with the Pardiso.jl library, which provides access to the PARDISO solver. 
Check out the [Pardiso.jl package](https://github.com/JuliaSparse/Pardiso.jl) for more details.

## Arguments

- `lib`: The library to use, either `:MKL` (for `Pardiso.MKLPardisoSolver`) or `:Pardiso` (for `Pardiso.PardisoSolver`). 
         Defaults to `:MKL`.
- `mtype`: The matrix type to use. Gets passed to the solver using `Pardiso.set_matrixtype!`.
          Defaults to `Pardiso.REAL_NONSYM`.
- `nthreads`: The number of threads to use for the solver. Gets passed to the solver using `Pardiso.set_nprocs!`.
              Defaults to `1`.
- `verbose`: If `true`, enables verbose output from the solver by setting the message level to `MESSAGE_LEVEL_ON`.
"""
function GridapSolvers.PardisoLinearSolver(;
  lib      = :MKL,
  mtype    = REAL_NONSYM,
  nthreads = 1,
  verbose  = false
)
  @assert lib in (:MKL,:Pardiso)
  @assert isa(mtype,MatrixType)
  ps = isequal(lib,:MKL) ? MKLPardisoSolver() : PardisoSolver()
  set_matrixtype!(ps, mtype)
  set_nprocs!(ps, nthreads)
  verbose && set_msglvl!(ps,MESSAGE_LEVEL_ON)
  solver = GridapPardisoSolver(ps,verbose)
  return finalizer(ps_finalize,solver)
end

function ps_finalize(solver)
  ps = solver.ps
  set_phase!(ps, RELEASE_ALL)
  pardiso(ps)
end

struct PardisoSymbolicSetup{A} <: Algebra.SymbolicSetup
  solver :: A
end

function Algebra.symbolic_setup(solver::GridapPardisoSolver, mat::AbstractMatrix)
  ps = solver.ps
  
  pardisoinit(ps)
  fix_iparm!(ps, :N)

  set_phase!(ps, ANALYSIS)
  ps_mat = ps_get_matrix(ps, mat, :N)
  ps_vec = allocate_in_domain(ps_mat)
  pardiso(ps, ps_mat, ps_vec)

  return PardisoSymbolicSetup(solver)
end

struct PardisoNumericalSetup{A,B} <: Algebra.NumericalSetup
  solver :: A
  ps_mat :: B
end

function Algebra.numerical_setup(ss::PardisoSymbolicSetup, mat::AbstractMatrix)
  ps = ss.solver.ps

  set_phase!(ps, NUM_FACT)

  ps_mat = ps_get_matrix(ps, mat, :N)
  ps_vec = allocate_in_domain(ps_mat)
  pardiso(ps, ps_mat, ps_vec)

  return PardisoNumericalSetup(ss.solver,ps_mat)
end

function Algebra.solve!(x::AbstractVector, ns::PardisoNumericalSetup, b::AbstractVector)
  ps = ns.solver.ps
  ps_mat = ns.ps_mat

  set_phase!(ps, SOLVE_ITERATIVE_REFINE)
  pardiso(ps, x, ps_mat, b)

  return x
end

end