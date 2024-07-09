
# TODO: This should be called NewtonRaphsonSolver, but it would clash with Gridap.
"""
    struct NewtonSolver <: Algebra.NonlinearSolver

  Newton-Raphson solver. Same as `NewtonRaphsonSolver` in Gridap, but with a couple addons:
  - Better logging and verbosity control.
  - Better convergence criteria.
  - Works with geometric LinearSolvers/Preconditioners.
"""
abstract type NonlinearSolverType end
struct GridapNonlinearSolver <: NonlinearSolverType end
struct NLSolveNonlinearSolver <: NonlinearSolverType end

struct NewtonSolver <: Algebra.NonlinearSolver
  ls ::Algebra.LinearSolver
  log::ConvergenceLog{Float64}
  type::NonlinearSolverType
end

nonlinear_solver_type(::NewtonSolver{T}) = T

function NewtonSolver(ls;maxiter=100,atol=1e-12,rtol=1.e-6,verbose=0,name="Newton-Raphson")
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols;verbose=verbose)
  return NewtonSolver(ls,log,GridapNonlinearSolver())
end

struct NewtonCache
  A::AbstractMatrix
  b::AbstractVector
  dx::AbstractVector
  ns::NumericalSetup
  # result
  # Inner constructor with default value for result
  # to have the same cache for NLSolve and Gridal Nonlinear solvers
  # NewtonCache(A::AbstractMatrix, b::AbstractVector, dx::AbstractVector, ns::NumericalSetup; result=nothing) = new(A, b, dx, ns, result)
end


function Algebra.solve!(x::AbstractVector,nls::NewtonSolver,op::NonlinearOperator,cache::Nothing)
  b  = residual(op, x)
  A  = jacobian(op, x)
  dx = allocate_in_domain(A); fill!(dx,zero(eltype(dx)))
  ss = symbolic_setup(nls.ls,A)
  ns = numerical_setup(ss,A,x)
  _solve_nr!(x,A,b,dx,ns,nls,op)
  return NewtonCache(A,b,dx,ns)
end

function Algebra.solve!(x::AbstractVector,nls::NewtonSolver,op::NonlinearOperator,cache::NewtonCache)
  A,b,dx,ns = cache.A, cache.b, cache.dx, cache.ns
  residual!(b, op, x)
  jacobian!(A, op, x)
  numerical_setup!(ns,A,x)
  _solve_nr!(x,A,b,dx,ns,nls,op)
  return cache
end

function _solve_nr!(x,A,b,dx,ns,nls,op)
  log = nls.log

  # Check for convergence on the initial residual
  res = norm(b)
  done = init!(log,res)

  # Newton-like iterations
  while !done

    # Solve linearized problem
    rmul!(b,-1)
    solve!(dx,ns,b)
    x .+= dx

    # Check convergence for the current residual
    residual!(b, op, x)
    res  = norm(b)
    done = update!(log,res)

    if !done
      # Update jacobian and solver
      jacobian!(A, op, x)
      numerical_setup!(ns,A,x)
    end

  end

  finalize!(log,res)
  return x
end

function NewtonSolver(ls;maxiter=100,atol=1e-12,rtol=1.e-6,verbose=0,name="Newton-Raphson")
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols;verbose=verbose)
  return NewtonSolver(ls,log,GridapNonlinearSolver())
end

# Same here, we are creating the same struct as NLSolver in Gridap
# but we cannot overwrite it. So we call it NLSolveNewtonSolver
# We should be able to use only one Newton, and merge the kwargs.
# Future work.
struct NLSolveNonlinearSolver <: NonlinearSolver
  ls::LinearSolver
  kwargs::Dict
  type::NonlinearSolverType
end

# I see result is also nothing (?)
# As commented about, we could merge the two types
mutable struct NLSolveNonlinearSolverCache <: GridapType
  f0::AbstractVector
  j0::AbstractMatrix
  df::OnceDifferentiable
  ns::NumericalSetup
  result
end

function NLSolveNonlinearSolver(;kwargs...)
  ls = BackslashSolver()
  Newton(ls,kwargs,NLSolveNonlinearSolver())
end

function NLSolveNonlinearSolver(ls::LinearSolver;kwargs...)
  @assert ! haskey(kwargs,:linsolve) "linsolve cannot be used here. It is managed internally"
  NLSolver(ls,kwargs,NLSolveNonlinearSolver())
end

function solve!(x::AbstractVector,nls::NLSolveNonlinearSolver,op::NonlinearOperator,cache::Nothing)
  cache = _new_nlsolve_cache(x,nls,op)
  _nlsolve_with_updated_cache!(x,nls,op,cache)
  cache
end

function solve!(
  x::AbstractVector,nls::NLSolveNonlinearSolver,op::NonlinearOperator,cache::NLSolveNonlinearSolverCache)
  cache = _update_nlsolve_cache!(cache,x,op)
  _nlsolve_with_updated_cache!(x,nls,op,cache)
  cache
end

function _nlsolve_with_updated_cache!(x,nls::NLSolveNonlinearSolver,op,cache)
  df = cache.df
  ns = cache.ns
  kwargs = nls.kwargs
  function linsolve!(x,A,b)
    numerical_setup!(ns,A,x)
    solve!(x,ns,b)
  end
  r = nlsolve(df,x;linsolve=linsolve!,kwargs...)
  cache.result = r
  copy_entries!(x,r.zero)
end

function _new_nlsolve_cache(x0,nls::NLSolveNonlinearSolver,op)
  f!(r,x) = residual!(r,op,x)
  j!(j,x) = jacobian!(j,op,x)
  fj!(r,j,x) = residual_and_jacobian!(r,j,op,x)
  f0, j0 = residual_and_jacobian(op,x0)
  df = OnceDifferentiable(f!,j!,fj!,x0,f0,j0)
  ss = symbolic_setup(nls.ls,j0)
  ns = numerical_setup(ss,j0,x0)
  NLSolveNonlinearSolverCache(f0,j0,df,ns,nothing)
end

function _update_nlsolve_cache!(cache,x0,op)
  f!(r,x) = residual!(r,op,x)
  j!(j,x) = jacobian!(j,op,x)
  fj!(r,j,x) = residual_and_jacobian!(r,j,op,x)
  f0 = cache.f0
  j0 = cache.j0
  ns = cache.ns
  residual_and_jacobian!(f0,j0,op,x0)
  df = OnceDifferentiable(f!,j!,fj!,x0,f0,j0)
  numerical_setup!(ns,j0,x0)
  NLSolveNonlinearSolverCache(f0,j0,df,ns,nothing)
end
