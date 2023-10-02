
mutable struct SolverTolerances{T <: Real}
  maxits :: Int
  atol   :: T
  rtol   :: T
  dtol   :: T
end

function SolverTolerances{T}(;maxits=1000, atol=eps(T), rtol=T(1.e-5), dtol=T(Inf)) where T
  return SolverTolerances{T}(maxits, atol, rtol, dtol)
end

get_solver_tolerances(s::Gridap.Algebra.LinearSolver) = @abstractmethod

function set_solver_tolerances!(a::SolverTolerances{T};
                                maxits = 1000,
                                atol   = eps(T),
                                rtol   = T(1.e-5),
                                dtol   = T(Inf)) where T
  a.maxits = maxits
  a.atol   = atol
  a.rtol   = rtol
  a.dtol   = dtol
  return a
end

function finished(tols::SolverTolerances,niter,e_r,e_a)
  return (niter > tols.maxits) || converged(tols,niter,e_r,e_a)
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
  println(io,"  - maxits: $(t.maxits)")
  println(io,"  - atol: $(t.atol)")
  println(io,"  - rtol: $(t.rtol)")
  println(io,"  - dtol: $(t.dtol)")
end

function Base.summary(t::SolverTolerances{T}) where T
  return "Tolerances: maxits=$(t.maxits), atol=$(t.atol), rtol=$(t.rtol), dtol=$(t.dtol)"
end
