
"""
    @enum SolverVerboseLevel begin
      SOLVER_VERBOSE_NONE = 0
      SOLVER_VERBOSE_LOW  = 1
      SOLVER_VERBOSE_HIGH = 2
    end

    SolverVerboseLevel(true) = SOLVER_VERBOSE_HIGH
    SolverVerboseLevel(false) = SOLVER_VERBOSE_NONE
"""
@enum SolverVerboseLevel begin
  SOLVER_VERBOSE_NONE = 0
  SOLVER_VERBOSE_LOW  = 1
  SOLVER_VERBOSE_HIGH = 2
end

SolverVerboseLevel(verbose::Bool) = (verbose ? SOLVER_VERBOSE_HIGH : SOLVER_VERBOSE_NONE)

"""
    mutable struct ConvergenceLog{T}
      ...
    end

    ConvergenceLog(
      name :: String,
      tols :: SolverTolerances{T};
      verbose = SOLVER_VERBOSE_NONE,
      depth   = 0
    )

  Standarized logging system for iterative linear solvers.

  # Methods:

  - [`reset!`](@ref)
  - [`init!`](@ref)
  - [`update!`](@ref)
  - [`finalize!`](@ref)
  - [`print_message`](@ref)
"""
mutable struct ConvergenceLog{T<:Real}
  name      :: String
  tols      :: SolverTolerances{T}
  num_iters :: Int
  residuals :: Vector{T}
  verbose   :: SolverVerboseLevel
  depth     :: Int
end

function ConvergenceLog(
  name :: String,
  tols :: SolverTolerances{T};
  verbose = SOLVER_VERBOSE_NONE,
  depth   = 0
) where T
  residuals = Vector{T}(undef,tols.maxiter+1)
  verbose = SolverVerboseLevel(verbose)
  return ConvergenceLog(name,tols,0,residuals,verbose,depth)
end

@inline get_tabulation(log::ConvergenceLog) = get_tabulation(log,2)
@inline get_tabulation(log::ConvergenceLog,n::Int) = repeat(' ', n + 2*log.depth)

"""
    set_depth!(log::ConvergenceLog,depth::Int)
    set_depth!(log::LinearSolver,depth::Int)

Sets the tabulation depth of the convergence log `log` to `depth`.
"""
function set_depth!(log::ConvergenceLog,depth::Int)
  log.depth = depth
  return log
end

function set_depth!(solver::Algebra.LinearSolver,depth::Int)
  if hasproperty(solver,:log)
    set_depth!(solver.log,depth)
  end
  map(children(solver)) do child
    set_depth!(child,depth)
  end
end

"""
    reset!(log::ConvergenceLog{T})

  Resets the convergence log `log` to its initial state.
"""
function reset!(log::ConvergenceLog{T}) where T
  log.num_iters = 0
  fill!(log.residuals,0.0)
  return log
end

"""
    init!(log::ConvergenceLog{T},r0::T)

  Initializes the convergence log `log` with the initial residual `r0`.
"""
function init!(log::ConvergenceLog{T},r0::T) where T
  log.num_iters = 0
  log.residuals[1] = r0
  if log.verbose > SOLVER_VERBOSE_LOW
    header =  " Starting $(log.name) solver "
    println(get_tabulation(log,0),rpad(string(repeat('-',15),header),55,'-'))
    t = get_tabulation(log)
    msg = @sprintf("> Iteration %3i - Residuals: %.2e,   %.2e ", 0, r0, 1)
    println(t,msg)
  end
  return finished(log.tols,log.num_iters,r0,1.0)
end

"""
    update!(log::ConvergenceLog{T},r::T)

  Updates the convergence log `log` with the residual `r` at the current iteration.
"""
function update!(log::ConvergenceLog{T},r::T) where T
  log.num_iters += 1
  log.residuals[log.num_iters+1] = r
  r_rel = r / log.residuals[1]
  if log.verbose > SOLVER_VERBOSE_LOW
    t = get_tabulation(log)
    msg = @sprintf("> Iteration %3i - Residuals: %.2e,   %.2e ", log.num_iters, r, r_rel)
    println(t,msg)
  end
  return finished(log.tols,log.num_iters,r,r_rel)
end

"""
    finalize!(log::ConvergenceLog{T},r::T)

  Finalizes the convergence log `log` with the final residual `r`.
"""
function finalize!(log::ConvergenceLog{T},r::T) where T
  r_rel = r / log.residuals[1]
  flag  = finished_flag(log.tols,log.num_iters,r,r_rel)
  if log.verbose > SOLVER_VERBOSE_NONE
    t = get_tabulation(log,0)
    println(t,"Solver $(log.name) finished with reason $(flag)")
    msg = @sprintf("Iterations: %3i - Residuals: %.2e,   %.2e ", log.num_iters, r, r_rel)
    println(t,msg)
    if log.verbose > SOLVER_VERBOSE_LOW
      footer =  " Exiting $(log.name) solver "
      println(t,rpad(string(repeat('-',15),footer),55,'-'))
    end
  end
  return flag
end

"""
    print_message(log::ConvergenceLog{T},msg::String)

  Prints the message `msg` to the output stream of the convergence log `log`.
"""
function print_message(log::ConvergenceLog{T},msg::String) where T
  if log.verbose > SOLVER_VERBOSE_LOW
    println(get_tabulation(log),msg)
  end
end

function Base.show(io::IO,k::MIME"text/plain",log::ConvergenceLog)
  println(io,"ConvergenceLog[$(log.name)]")
  println(io," > tols: $(summary(log.tols))")
  println(io," > num_iter: $(log.num_iters)")
  println(io," > residual: $(log.residuals[log.num_iters+1])")
end

function Base.summary(log::ConvergenceLog)
  r_abs = log.residuals[log.num_iters+1]
  r_rel = r_abs / log.residuals[1]
  flag  = finished_flag(log.tols,log.num_iters,r_abs,r_rel)
  msg   = "Convergence[$(log.name)]: conv_flag=$(flag), niter=$(log.num_iters), r_abs=$(r_abs), r_rel=$(r_rel)"
  return msg
end
