
struct LinearSolverFromSmoother{A} <: Algebra.LinearSolver
  smoother :: A
end

struct LinearSolverFromSmootherSS{A,B} <: Algebra.SymbolicSetup
  smoother :: A
  smoother_ss :: B
end

struct LinearSolverFromSmootherNS{A,B,C} <: Algebra.NumericalSetup
  smoother    :: A
  smoother_ns :: B
  caches      :: C
end

function Gridap.Algebra.symbolic_setup(solver::LinearSolverFromSmoother, mat::AbstractMatrix)
  ss = symbolic_setup(solver.smoother,mat)
  return LinearSolverFromSmootherSS(solver.smoother,ss)
end

function Gridap.Algebra.numerical_setup(ss::LinearSolverFromSmootherSS, mat::AbstractMatrix)
  ns = numerical_setup(ss.smoother_ss, mat)
  caches = allocate_in_domain(mat)
  return LinearSolverFromSmootherNS(ss.smoother,ns,caches)
end

function Gridap.Algebra.numerical_setup(ss::LinearSolverFromSmootherSS, mat::AbstractMatrix, vec::AbstractVector)
  ns = numerical_setup(ss.smoother_ss, mat, vec)
  caches = allocate_in_domain(mat)
  return LinearSolverFromSmootherNS(ss.smoother,ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::LinearSolverFromSmootherNS, mat::AbstractMatrix)
  numerical_setup!(ns.smoother_ns, mat)
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::LinearSolverFromSmootherNS, mat::AbstractMatrix, vec::AbstractVector)
  numerical_setup!(ns.smoother_ns, mat, vec)
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::LinearSolverFromSmootherNS, b::AbstractVector)
  r = ns.caches
  fill!(x,zero(eltype(x)))
  copy!(r,b)
  solve!(x,ns.smoother_ns,r)
  return x
end
