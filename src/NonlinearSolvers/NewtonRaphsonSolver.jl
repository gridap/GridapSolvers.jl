
# TODO: This should be called NewtonRaphsonSolver, but it would clash with Gridap. 
struct NewtonSolver <: Algebra.NonlinearSolver
  ls ::Algebra.LinearSolver
  log::ConvergenceLog{Float64}
end

function NewtonSolver(ls;maxiter=100,atol=1e-12,rtol=1.e-6,verbose=0,name="Newton-Raphson")
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols;verbose=verbose)
  return NewtonSolver(ls,log)
end

struct NewtonCache
  A::AbstractMatrix
  b::AbstractVector
  dx::AbstractVector
  ns::NumericalSetup
end

function Algebra.solve!(x::AbstractVector,nls::NewtonSolver,op::NonlinearOperator,cache::Nothing)
  b  = residual(op, x)
  A  = jacobian(op, x)
  dx = similar(b)
  ss = symbolic_setup(nls.ls, A)
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
