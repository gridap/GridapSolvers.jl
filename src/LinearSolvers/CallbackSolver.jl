
struct CallbackSolver{A,B} <: Algebra.LinearSolver
  solver :: A
  callback :: B
  
  function CallbackSolver(solver::LinearSolver,callback::Function)
    A = typeof(solver)
    B = typeof(callback)
    new{A,B}(solver,callback)
  end
end

struct CallbackSolverSS{A,B} <: Algebra.SymbolicSetup
  solver :: A
  ss :: B
end

function Algebra.symbolic_setup(solver::CallbackSolver,mat::AbstractMatrix)
  ss = Algebra.symbolic_setup(solver.solver,mat)
  return CallbackSolverSS(solver,ss)
end

struct CallbackSolverNS{A,B} <: Algebra.NumericalSetup
  solver :: A
  ns :: B
end

function Algebra.numerical_setup(ss::CallbackSolverSS,mat::AbstractMatrix)
  ns = Algebra.numerical_setup(ss.ss,mat)
  return CallbackSolverNS(ss.solver,ns)
end

function Algebra.numerical_setup(ss::CallbackSolverSS,mat::AbstractMatrix,x::AbstractVector)
  ns = Algebra.numerical_setup(ss.ss,mat,x)
  return CallbackSolverNS(ss.solver,ns)
end

function Algebra.numerical_setup!(ns::CallbackSolverNS,mat::AbstractMatrix)
  Algebra.numerical_setup!(ns.ns,mat)
  return ns
end

function Algebra.numerical_setup!(ns::CallbackSolverNS,mat::AbstractMatrix,x::AbstractVector)
  Algebra.numerical_setup!(ns.ns,mat,x)
  return ns
end

function Algebra.solve!(x::AbstractVector,ns::CallbackSolverNS,b::AbstractVector)
  solve!(x,ns.ns,b)
  ns.solver.callback(x)
  return x
end
