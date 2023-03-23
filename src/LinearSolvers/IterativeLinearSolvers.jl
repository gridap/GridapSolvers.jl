
abstract type IterativeLinearSolverType end
struct CGIterativeSolverType     <: IterativeLinearSolverType end
struct GMRESIterativeSolverType  <: IterativeLinearSolverType end
struct MINRESIterativeSolverType <: IterativeLinearSolverType end

# Solvers

"""
  Wrappers for [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
  krylov-like iterative solvers.  
"""
struct IterativeLinearSolver{A} <: Gridap.Algebra.LinearSolver
  kwargs

  function IterativeLinearSolver(type::IterativeLinearSolverType,kwargs)
    A = typeof(type)
    return new{A}(kwargs)
  end
end

SolverType(::IterativeLinearSolver{T}) where T = T()

function ConjugateGradientSolver(;kwargs...)
  options = [:statevars,:initially_zero,:Pl,:abstol,:reltol,:maxiter,:verbose,:log]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  return IterativeLinearSolver(CGIterativeSolverType(),kwargs)
end

function GMRESSolver(;kwargs...)
  options = [:initially_zero,:abstol,:reltol,:restart,:maxiter,:Pl,:Pr,:log,:verbose,:orth_meth]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  return IterativeLinearSolver(GMRESIterativeSolverType(),kwargs)
end

function MINRESSolver(;kwargs...)
  options = [:initially_zero,:skew_hermitian,:abstol,:reltol,:maxiter,:log,:verbose]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  return IterativeLinearSolver(MINRESIterativeSolverType(),kwargs)
end

# Symbolic setup

struct IterativeLinearSolverSS <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::IterativeLinearSolver,A::AbstractMatrix)
  IterativeLinearSolverSS(solver)
end

# Numerical setup

struct IterativeLinearSolverNS <: Gridap.Algebra.NumericalSetup
  solver
  A
end

function Gridap.Algebra.numerical_setup(ss::IterativeLinearSolverSS,A::AbstractMatrix)
  IterativeLinearSolverNS(ss.solver,A)
end

function Gridap.Algebra.solve!(x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  solver_type = SolverType(ns.solver)
  solve!(solver_type,x,ns,y)
end

function(::IterativeLinearSolverType,::AbstractVector,::IterativeLinearSolverNS,::AbstractVector)
  @abstractmethod
end

function Gridap.Algebra.solve!(::CGIterativeSolverType,
                               x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  A, kwargs = ns.A, ns.solver.kwargs
  return cg!(x,A,y;kwargs...)
end

function Gridap.Algebra.solve!(::GMRESIterativeSolverType,
                               x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  A, kwargs = ns.A, ns.solver.kwargs
  return gmres!(x,A,y;kwargs...)
end

function Gridap.Algebra.solve!(::MINRESIterativeSolverType,
                               x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  A, kwargs = ns.A, ns.solver.kwargs
  return minres!(x,A,y;kwargs...)
end