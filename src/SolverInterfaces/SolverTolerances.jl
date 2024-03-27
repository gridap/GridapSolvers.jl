"""
    @enum SolverConvergenceFlag begin
      SOLVER_CONVERGED_ATOL     = 0
      SOLVER_CONVERGED_RTOL     = 1
      SOLVER_DIVERGED_MAXITER   = 2
      SOLVER_DIVERGED_BREAKDOWN = 3
    end

  Convergence flags for iterative linear solvers.
"""
@enum SolverConvergenceFlag begin
  SOLVER_CONVERGED_ATOL     = 0
  SOLVER_CONVERGED_RTOL     = 1
  SOLVER_DIVERGED_MAXITER   = 2
  SOLVER_DIVERGED_BREAKDOWN = 3
end

"""
    mutable struct SolverTolerances{T}
      ...
    end

    SolverTolerances{T}(
      maxiter :: Int = 1000,
      atol    :: T   = eps(T),
      rtol    :: T   = 1.e-5,
      dtol    :: T   = Inf
    )

  Structure to check convergence conditions for iterative linear solvers.

  # Methods:

  - [`get_solver_tolerances`](@ref)
  - [`set_solver_tolerances!`](@ref)
  - [`converged`](@ref)
  - [`finished`](@ref)
  - [`finished_flag`](@ref)
"""
mutable struct SolverTolerances{T <: Real}
  maxiter :: Int
  atol    :: T
  rtol    :: T
  dtol    :: T
end

function SolverTolerances{T}(;maxiter=1000, atol=eps(T), rtol=T(1.e-5), dtol=T(Inf)) where T
  return SolverTolerances{T}(maxiter, atol, rtol, dtol)
end

"""
    get_solver_tolerances(s::LinearSolver)

  Returns the solver tolerances of the linear solver `s`.
"""
get_solver_tolerances(s::Gridap.Algebra.LinearSolver) = @abstractmethod

"""
    set_solver_tolerances!(s::LinearSolver;
      maxiter = 1000,
      atol   = eps(T),
      rtol   = T(1.e-5),
      dtol   = T(Inf)
    )

  Modifies tolerances of the linear solver `s`.
"""
function set_solver_tolerances!(s::Gridap.Algebra.LinearSolver;kwargs...) 
  set_solver_tolerances!(get_solver_tolerances(s);kwargs...)
end

function set_solver_tolerances!(a::SolverTolerances{T};
                                maxiter = 1000,
                                atol   = eps(T),
                                rtol   = T(1.e-5),
                                dtol   = T(Inf)) where T
  a.maxiter = maxiter
  a.atol = atol
  a.rtol = rtol
  a.dtol = dtol
  return a
end

"""
    finished_flag(tols::SolverTolerances,niter,e_a,e_r) :: SolverConvergenceFlag

  Computes the solver exit condition given 

   - the number of iterations `niter`
   - the absolute error `e_a` 
   - and the relative error `e_r`.

  Returns the corresponding `SolverConvergenceFlag`.
"""
function finished_flag(tols::SolverTolerances,niter,e_a,e_r) :: SolverConvergenceFlag
  if !finished(tols,niter,e_a,e_r)
    @warn "finished_flag() called with unfinished solver!"
  end
  if e_r < tols.rtol
    return SOLVER_CONVERGED_RTOL
  elseif e_a < tols.atol
    return SOLVER_CONVERGED_ATOL
  elseif niter >= tols.maxiter
    return SOLVER_DIVERGED_MAXITER
  else
    return SOLVER_DIVERGED_BREAKDOWN
  end
end

"""
    finished(tols::SolverTolerances,niter,e_a,e_r) :: Bool

  Returns `true` if the solver has finished, `false` otherwise.
"""
function finished(tols::SolverTolerances,niter,e_a,e_r) :: Bool
  return (niter >= tols.maxiter) || converged(tols,niter,e_a,e_r)
end

"""
    converged(tols::SolverTolerances,niter,e_a,e_r) :: Bool

  Returns `true` if the solver has converged, `false` otherwise.
"""
function converged(tols::SolverTolerances,niter,e_a,e_r) :: Bool
  return (e_r < tols.rtol) || (e_a < tols.atol)
end

function Base.show(io::IO,k::MIME"text/plain",t::SolverTolerances{T}) where T
  println(io,"SolverTolerances{$T}:")
  println(io,"  - maxiter: $(t.maxiter)")
  println(io,"  - atol: $(t.atol)")
  println(io,"  - rtol: $(t.rtol)")
  println(io,"  - dtol: $(t.dtol)")
end

function Base.summary(t::SolverTolerances{T}) where T
  return "Tolerances: maxiter=$(t.maxiter), atol=$(t.atol), rtol=$(t.rtol), dtol=$(t.dtol)"
end
