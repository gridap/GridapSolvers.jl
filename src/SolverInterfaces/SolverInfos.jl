

"""
ConvergenceLog
ConvergenceTracker
ConvergenceMonitor

"""

struct SolverInfo{T<:Real}
  name :: String
  tols :: SolverTolerances{T}
  data :: Dict{Symbol, Any}
end

SolverInfo(name::String) = SolverInfo(name,SolverTolerances{Float64}())
SolverInfo(name::String,tols::SolverTolerances) = SolverInfo(name,tols,Dict{Symbol, Any}())

function get_solver_info(::Gridap.Algebra.LinearSolver)
  @abstractmethod
end

function log_info!(a::SolverInfo,key::Symbol,val)
  a.data[key] = val
end

function log_iteration_info!(a::SolverInfo,key::Symbol,val::T) where T
  log_key = Symbol("log_",key)
  if !haskey(a.data, log_key)
    a.data[log_key] = Vector{T}()
  end
  push!(a.data[log_key], val)
end

function log_convergence_info!(a::SolverInfo{T}, niter::Int, e_rel::T, e_abs::T) where T
  tols = a.tols
  if e_abs < tols.atol
    flag = SOLVER_CONVERGED_ATOL
  elseif e_rel < tols.rtol
    flag = SOLVER_CONVERGED_RTOL
  elseif niter >= tols.maxits
    flag = SOLVER_DIVERGED_MAXITER
  else # We have stopped because of a breakdown
    flag = SOLVER_DIVERGED_BREAKDOWN
  end
  log_info!(a,:convergence_flag,flag)
  log_info!(a,:num_iters,niter)
  log_info!(a,:err_rel,e_rel)
  log_info!(a,:err_abs,e_abs)
  return a
end

function log_iteration_error!(a::SolverInfo{T}, e_rel::T, e_abs::T) where T
  log_iteration_info!(a,:err_rel,e_rel)
  log_iteration_info!(a,:err_abs,e_abs)
end

function Base.show(io::IO,k::MIME"text/plain",a::SolverInfo)
  println(io,"SolverInfo[$(a.name)]")
  println(io," > " * summary(a.tols))

  d = a.data
  if haskey(d,:convergence_flag)
    println(io," > Convergence:")
    println(io,"  - conv flag: $(d[:convergence_flag])")
    println(io,"  - num iters: $(d[:num_iters])")
    println(io,"  - rel error: $(d[:err_rel])")
    println(io,"  - abs error: $(d[:err_abs])")
  else
    println(io," > Convergence not set.")
  end
end

function Base.summary(a::SolverInfo) 
  msg = "[$(a.name)]["
  if haskey(a.data,:convergence_flag)
    msg *= "conv_flag=$(a.data[:convergence_flag]), "
    msg *= "niter=$(a.data[:num_iters]), "
    msg *= "err_rel=$(a.data[:err_rel]), "
    msg *= "err_abs=$(a.data[:err_abs])"
  end
  msg *= "]"
  return msg
end
