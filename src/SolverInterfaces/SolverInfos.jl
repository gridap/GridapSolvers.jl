
struct SolverInfo{T<:Real}
  name :: String
  data :: Dict{Symbol, Any}
end

SolverInfo(name::String) = SolverInfo(name,Dict{Symbol, Any}())

function get_solver_info(::Gridap.Algebra.LinearSolver)
  return SolverInfo("Empty solver info")
end

function add_info!(a::SolverInfo,key::Union{Symbol,String},val;prefix="")
  key = Symbol(prefix,key)
  a.data[key] = val
end

function add_convergence_info!(a::SolverInfo,log::ConvergenceLog;prefix="")
  prefix = string(prefix,log.name)
  add_info!(a,:num_iters,log.num_iters,prefix=prefix)
  add_info!(a,:residuals,log.residuals,prefix=prefix)
end

function add_tolerance_info!(a::SolverInfo,tols::SolverTolerances;prefix="")
  add_info!(a,:maxiter,tols.maxiter,prefix=prefix)
  add_info!(a,:atol,tols.atol,prefix=prefix)
  add_info!(a,:rtol,tols.rtol,prefix=prefix)
end

function add_tolerance_info!(a::SolverInfo,log::ConvergenceLog;prefix="")
  prefix = string(prefix,log.name)
  add_tolerance_info!(a,log.tols,prefix=prefix)
end
