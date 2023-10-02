
mutable struct ConvergenceLog{T <: Real}
  name      :: String
  tols      :: SolverTolerances{T}
  num_iters :: Int
  residuals :: Vector{T}
  verbose   :: SolverVerboseLevel
end

function ConvergenceLog(name::String,tols::SolverTolerances{T},verbose)
  residuals = Vector{T}(undef,tols.maxits+1)
  return ConvergenceLog(name,tols,0,residuals,verbose)
end

function ConvergenceLog(name::String,tols::SolverTolerances{T}) where T
  return ConvergenceLog(name,tols,SOLVER_VERBOSE_NONE)
end

function reset!(log::ConvergenceLog{T}) where T
  log.num_iters = 0
  fill!(log.residuals,0.0)
  return log
end

function init!(log::ConvergenceLog{T},r0::T) where T
  log.num_iters = 0
  log.residuals[1] = r0
  if log.verbose > SOLVER_VERBOSE_NONE
    println(" > Starting $(log.name) solver:")
    if log.verbose > SOLVER_VERBOSE_LOW
      println("   > Iteration 0 - Residual: $(r0)")
    end
  end
  return finished(log.tols,log.num_iters,r0,1.0)
end

function update!(log::ConvergenceLog{T},r::T) where T
  log.num_iters += 1
  log.residuals[log.num_iters+1] = r
  if log.verbose > SOLVER_VERBOSE_LOW
    println("   > Iteration $(log.num_iters) - Residual: $(r)")
  end
  r_rel = r / log.residuals[1]
  return finished(log.tols,log.num_iters,r,r_rel)
end

function finalize!(log::ConvergenceLog{T}) where T
  log.num_iters += 1
  log.residuals[log.num_iters+1] = r

  r_rel = r / log.residuals[1]
  flag  = convergence_reason(log.tols,log.num_iters,r,r_rel)
  if log.verbose > SOLVER_VERBOSE_NONE
    println(" >  Solver $(log.name) finished with reason $(flag)")
    println("      Num Iterations: $(log.num_iters) - Residual: $(r)")
  end
  return flag
end
