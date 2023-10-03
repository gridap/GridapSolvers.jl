@enum SolverConvergenceFlag begin
  SOLVER_CONVERGED_ATOL     = 0
  SOLVER_CONVERGED_RTOL     = 1
  SOLVER_DIVERGED_MAXITER   = 2
  SOLVER_DIVERGED_BREAKDOWN = 3
end

mutable struct SolverTolerances{T <: Real}
  maxiter :: Int
  atol    :: T
  rtol    :: T
  dtol    :: T
end

function SolverTolerances{T}(;maxiter=1000, atol=eps(T), rtol=T(1.e-5), dtol=T(Inf)) where T
  return SolverTolerances{T}(maxiter, atol, rtol, dtol)
end

get_solver_tolerances(s::Gridap.Algebra.LinearSolver) = @abstractmethod

function set_solver_tolerances!(a::SolverTolerances{T};
                                maxiter = 1000,
                                atol   = eps(T),
                                rtol   = T(1.e-5),
                                dtol   = T(Inf)) where T
  a.maxiter = maxiter
  a.atol   = atol
  a.rtol   = rtol
  a.dtol   = dtol
  return a
end

function finished_flag(tols::SolverTolerances,niter,e_r,e_a)
  if !finished(tols,niter,e_r,e_a)
    @warn "finished_flag() called with unfinished solver!"
  end
  if niter > tols.maxiter
    return SOLVER_DIVERGED_MAXITER
  elseif e_r < tols.rtol
    return SOLVER_CONVERGED_RTOL
  elseif e_a < tols.atol
    return SOLVER_CONVERGED_ATOL
  else
    return SOLVER_DIVERGED_BREAKDOWN
  end
end

function finished(tols::SolverTolerances,niter,e_r,e_a)
  return (niter >= tols.maxiter) || converged(tols,niter,e_r,e_a)
end

function converged(tols::SolverTolerances,niter,e_r,e_a)
  return (e_r < tols.rtol) || (e_a < tols.atol)
end

function set_solver_tolerances!(s::Gridap.Algebra.LinearSolver;kwargs...)
  a = get_solver_tolerances(s)
  return set_solver_tolerances!(a;kwargs...)
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
