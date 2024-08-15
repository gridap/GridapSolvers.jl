
struct MatrixSolver <: Algebra.LinearSolver
  M::AbstractMatrix
  solver::Algebra.LinearSolver
  function MatrixSolver(M;solver=LUSolver())
    new(M,solver)
  end
end

struct MatrixSolverSS <: Algebra.SymbolicSetup
  solver::MatrixSolver
  ss::Algebra.SymbolicSetup

  function MatrixSolverSS(solver::MatrixSolver)
    ss = Algebra.symbolic_setup(solver.solver, solver.M)
    new(solver, ss)
  end
end

Algebra.symbolic_setup(solver::MatrixSolver,mat::AbstractMatrix) = MatrixSolverSS(solver)

struct MatrixSolverNS <: Algebra.NumericalSetup
  solver::MatrixSolver
  ns::Algebra.NumericalSetup

  function MatrixSolverNS(ss::MatrixSolverSS)
    solver = ss.solver
    ns = Gridap.Algebra.numerical_setup(ss.ss, solver.M)
    new(solver, ns)
  end
end

Algebra.numerical_setup(ss::MatrixSolverSS,mat::AbstractMatrix) = MatrixSolverNS(ss)

function Algebra.solve!(x::AbstractVector, solver::MatrixSolverNS, b::AbstractVector)
  Algebra.solve!(x, solver.ns, b)
end
