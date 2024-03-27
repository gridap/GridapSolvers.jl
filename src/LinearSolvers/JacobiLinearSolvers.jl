"""
    struct JacobiLinearSolver <: Gridap.Algebra.LinearSolver

  Given a matrix `A`, the Jacobi or Diagonal preconditioner is defined as `P = diag(A)`.
"""
struct JacobiLinearSolver <: Gridap.Algebra.LinearSolver
end

struct JacobiSymbolicSetup <: Gridap.Algebra.SymbolicSetup
end

function Gridap.Algebra.symbolic_setup(s::JacobiLinearSolver,A::AbstractMatrix)
  JacobiSymbolicSetup()
end

mutable struct JacobiNumericalSetup{A} <: Gridap.Algebra.NumericalSetup
  inv_diag :: A
end

function Gridap.Algebra.numerical_setup(ss::JacobiSymbolicSetup,A::AbstractMatrix)
  inv_diag = 1.0./diag(A)
  return JacobiNumericalSetup(inv_diag)
end

function Gridap.Algebra.numerical_setup!(ns::JacobiNumericalSetup, A::AbstractMatrix)
  ns.inv_diag .= 1.0 ./ diag(a)
end

function Gridap.Algebra.numerical_setup(ss::JacobiSymbolicSetup,A::PSparseMatrix)
  inv_diag = map(own_values(A)) do a
    1.0 ./ diag(a)
  end
  return JacobiNumericalSetup(inv_diag)
end

function Gridap.Algebra.numerical_setup!(ns::JacobiNumericalSetup, A::PSparseMatrix)
  map(ns.inv_diag,own_values(A)) do inv_diag, a
    inv_diag .= 1.0 ./ diag(a)
  end
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector, ns::JacobiNumericalSetup, b::AbstractVector)
  inv_diag = ns.inv_diag
  x .= inv_diag .* b
  return x
end

function Gridap.Algebra.solve!(x::PVector, ns::JacobiNumericalSetup, b::PVector)
  inv_diag = ns.inv_diag
  map(inv_diag,own_values(x),own_values(b)) do inv_diag, x, b
    x .= inv_diag .* b
  end
  return x
end

function LinearAlgebra.ldiv!(x::AbstractVector,ns::JacobiNumericalSetup,b::AbstractVector)
  solve!(x,ns,b)
end