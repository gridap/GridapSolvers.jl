
# Identity solver, for testing purposes 
struct IdentitySolver <: Gridap.Algebra.LinearSolver
end

struct IdentitySymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(s::IdentitySolver,A::AbstractMatrix) 
  IdentitySymbolicSetup(s)
end

struct IdentityNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
end

function Gridap.Algebra.numerical_setup(ss::IdentitySymbolicSetup,mat::AbstractMatrix)
  s = ss.solver
  return IdentityNumericalSetup(s)
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::IdentityNumericalSetup,y::AbstractVector)
  copy!(x,y)
  return x
end
