
"""
    NLsolveNonlinearSolver <: NonlinearSolver

    NLsolveNonlinearSolver(ls::LinearSolver;kwargs...)
    NLsolveNonlinearSolver(;kwargs...)

Wrapper for `NLsolve.jl` nonlinear solvers. It is equivalent to the wrappers in Gridap, but 
with support for nonlinear preconditioners. Same `kwargs` as in `nlsolve`.
Due to `NLSolve.jl` not using LinearAlgebra's API, these solvers are not compatible with
`PartitionedArrays.jl`. For parallel computations, use [`NewtonSolver`](@ref) instead.
"""
struct NLsolveNonlinearSolver{A} <: NonlinearSolver
  ls::A
  kwargs::Dict
  function NLsolveNonlinearSolver(
    ls::LinearSolver;kwargs...
  )
    @assert ! haskey(kwargs,:linsolve) "linsolve cannot be used here. It is managed internally"
    A = typeof(ls)
    new{A}(ls,kwargs)
  end
end

function NLsolveNonlinearSolver(;kwargs...)
  ls = LUSolver()
  NLsolveNonlinearSolver(ls;kwargs...)
end

mutable struct NLSolversCache <: GridapType
  f0::AbstractVector
  j0::AbstractMatrix
  df::OnceDifferentiable
  ns::NumericalSetup
  cache
  result
end

function Algebra.solve!(
  x::AbstractVector,nls::NLsolveNonlinearSolver,op::NonlinearOperator,cache::Nothing
)
  cache = _new_nlsolve_cache(x,nls,op)
  _nlsolve_with_updated_cache!(x,nls,op,cache)
  cache
end

function Algebra.solve!(
  x::AbstractVector,nls::NLsolveNonlinearSolver,op::NonlinearOperator,cache::NLSolversCache
)
  cache = _update_nlsolve_cache!(cache,x,op)
  _nlsolve_with_updated_cache!(x,nls,op,cache)
  cache
end

function _nlsolve_with_updated_cache!(x,nls::NLsolveNonlinearSolver,op,cache)
  df = cache.df
  ns = cache.ns
  kwargs = nls.kwargs
  internal_cache = cache.cache
  function linsolve!(p,A,b)
    fill!(p,zero(eltype(p))) # Needed because NLSolve gives you NaNs on the 1st iteration...
    numerical_setup!(ns,A,internal_cache.x)
    solve!(p,ns,b)
  end
  r = _nlsolve(df,x;linsolve=linsolve!,cache=internal_cache,kwargs...)
  cache.result = r
  copy_entries!(x,r.zero)
end

function _new_nlsolve_cache(x0,nls::NLsolveNonlinearSolver,op)
  f!(r,x) = residual!(r,op,x)
  j!(j,x) = jacobian!(j,op,x)
  fj!(r,j,x) = residual_and_jacobian!(r,j,op,x)
  f0, j0 = residual_and_jacobian(op,x0)
  df = OnceDifferentiable(f!,j!,fj!,x0,f0,j0)
  ss = symbolic_setup(nls.ls,j0,x0)
  ns = numerical_setup(ss,j0,x0)
  cache = nlsolve_internal_cache(df,nls.kwargs)
  NLSolversCache(f0,j0,df,ns,cache,nothing)
end

function _update_nlsolve_cache!(cache,x0,op)
  f!(r,x) = residual!(r,op,x)
  j!(j,x) = jacobian!(j,op,x)
  fj!(r,j,x) = residual_and_jacobian!(r,j,op,x)
  f0 = cache.f0
  j0 = cache.j0
  ns = cache.ns
  residual_and_jacobian!(f0,j0,op,x0)
  df = NLsolve.OnceDifferentiable(f!,j!,fj!,x0,f0,j0)
  numerical_setup!(ns,j0,x0)
  NLSolversCache(f0,j0,df,ns,cache.cache,nothing)
end

nlsolve_internal_cache(df,kwargs) = nlsolve_internal_cache(Val(kwargs[:method]),df,kwargs)
nlsolve_internal_cache(::Val{:newton},df,kwargs) = NLsolve.NewtonCache(df)
nlsolve_internal_cache(::Val{:trust_region},df,kwargs) = NLsolve.NewtonTrustRegionCache(df)
nlsolve_internal_cache(::Val{:anderson},df,kwargs) = NLsolve.AndersonCache(df,kwargs[:m])

# Copied from https://github.com/JuliaNLSolvers/NLsolve.jl/blob/master/src/nlsolve/nlsolve.jl
# We need to modify it so that we can access the internal caches...
function _nlsolve(
  df::Union{NLsolve.NonDifferentiable, NLsolve.OnceDifferentiable},
  initial_x::AbstractArray;
  method::Symbol = :trust_region,
  xtol::Real = zero(real(eltype(initial_x))),
  ftol::Real = convert(real(eltype(initial_x)), 1e-8),
  iterations::Integer = 1_000,
  store_trace::Bool = false,
  show_trace::Bool = false,
  extended_trace::Bool = false,
  linesearch = LineSearches.Static(),
  linsolve=(x, A, b) -> copyto!(x, A\b),
  factor::Real = one(real(eltype(initial_x))),
  autoscale::Bool = true,
  m::Integer = 10,
  beta::Real = 1,
  aa_start::Integer = 1,
  droptol::Real = convert(real(eltype(initial_x)), 1e10),
  cache = nothing
)
  if show_trace
    println("Iter     f(x) inf-norm    Step 2-norm")
    println("------   --------------   --------------")
  end
  if method == :newton
    NLsolve.newton_(
      df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace, linesearch, linsolve, cache
    )
  elseif method == :trust_region
    NLsolve.trust_region_(
      df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace, factor, autoscale, cache
    )
  elseif method == :anderson
    NLsolve.anderson(
      df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace, beta, aa_start, droptol, cache
    )
  elseif method == :broyden
    NLsolve.broyden(
      df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace, linesearch
    )
  else
    throw(ArgumentError("Unknown method $method"))
  end
end
