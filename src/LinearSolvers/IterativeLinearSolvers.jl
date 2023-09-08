
abstract type IterativeLinearSolverType end
struct CGIterativeSolverType     <: IterativeLinearSolverType end
struct GMRESIterativeSolverType  <: IterativeLinearSolverType end
struct MINRESIterativeSolverType <: IterativeLinearSolverType end
struct SSORIterativeSolverType   <: IterativeLinearSolverType end

# Constructors

"""
  Wrappers for [IterativeSolvers.jl](https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl)
  krylov-like iterative solvers.

  Currently supported: 
  - ConjugateGradientSolver
  - GMRESSolver
  - MINRESSolver
"""
struct IterativeLinearSolver{A} <: Gridap.Algebra.LinearSolver
  args
  kwargs

  function IterativeLinearSolver(type::IterativeLinearSolverType,args,kwargs)
    A = typeof(type)
    return new{A}(args,kwargs)
  end
end

SolverType(::IterativeLinearSolver{T}) where T = T()

function IS_ConjugateGradientSolver(;kwargs...)
  options = [:statevars,:initially_zero,:Pl,:abstol,:reltol,:maxiter,:verbose,:log]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  return IterativeLinearSolver(CGIterativeSolverType(),nothing,kwargs)
end

function IS_GMRESSolver(;kwargs...)
  options = [:initially_zero,:abstol,:reltol,:restart,:maxiter,:Pl,:Pr,:log,:verbose,:orth_meth]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  return IterativeLinearSolver(GMRESIterativeSolverType(),nothing,kwargs)
end

function IS_MINRESSolver(;kwargs...)
  options = [:initially_zero,:skew_hermitian,:abstol,:reltol,:maxiter,:log,:verbose]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  return IterativeLinearSolver(MINRESIterativeSolverType(),nothing,kwargs)
end

function IS_SSORSolver(ω::Real;kwargs...)
  options = [:maxiter]
  @check all(map(opt -> opt ∈ options,keys(kwargs)))
  args = Dict(:ω => ω)
  return IterativeLinearSolver(SSORIterativeSolverType(),args,kwargs)
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
  caches
end

function Gridap.Algebra.numerical_setup(ss::IterativeLinearSolverSS,A::AbstractMatrix)
  solver_type = SolverType(ss.solver)
  numerical_setup(solver_type,ss,A)
end

function Gridap.Algebra.numerical_setup(::IterativeLinearSolverType,
                                        ss::IterativeLinearSolverSS,
                                        A::AbstractMatrix)
  IterativeLinearSolverNS(ss.solver,A,nothing)
end

function Gridap.Algebra.numerical_setup(::CGIterativeSolverType,
                                        ss::IterativeLinearSolverSS,
                                        A::AbstractMatrix)
  x = allocate_col_vector(A)
  caches = IterativeSolvers.CGStateVariables(zero(x), similar(x), similar(x))
  return IterativeLinearSolverNS(ss.solver,A,caches)
end

function Gridap.Algebra.numerical_setup(::SSORIterativeSolverType,
                                        ss::IterativeLinearSolverSS,
                                        A::AbstractMatrix)
  x = allocate_row_vector(A)
  b = allocate_col_vector(A)
  ω       = ss.solver.args[:ω]
  maxiter = ss.solver.kwargs[:maxiter]
  caches  = IterativeSolvers.ssor_iterable(x,A,b,ω;maxiter=maxiter)
  return IterativeLinearSolverNS(ss.solver,A,caches)
end

function IterativeSolvers.ssor_iterable(x::PVector,
                                        A::PSparseMatrix,
                                        b::PVector,
                                        ω::Real;
                                        maxiter::Int = 10)
  iterables = map(own_values(x),own_values(A),own_values(b)) do _xi,_Aii,_bi
    xi  = Vector(_xi)
    Aii = SparseMatrixCSC(_Aii)
    bi  = Vector(_bi)
    return IterativeSolvers.ssor_iterable(xi,Aii,bi,ω;maxiter=maxiter)
  end
  return iterables
end

# Solve

function LinearAlgebra.ldiv!(x::AbstractVector,ns::IterativeLinearSolverNS,b::AbstractVector)
  solve!(x,ns,b)
end

function Gridap.Algebra.solve!(x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  solver_type = SolverType(ns.solver)
  solve!(solver_type,x,ns,y)
end

function Gridap.Algebra.solve!(::IterativeLinearSolverType,
                               ::AbstractVector,
                               ::IterativeLinearSolverNS,
                               ::AbstractVector)
  @abstractmethod
end

function Gridap.Algebra.solve!(::CGIterativeSolverType,
                               x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  A, kwargs, caches = ns.A, ns.solver.kwargs, ns.caches
  return cg!(x,A,y;kwargs...,statevars=caches)
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

function Gridap.Algebra.solve!(::SSORIterativeSolverType,
                               x::AbstractVector,
                               ns::IterativeLinearSolverNS,
                               y::AbstractVector)
  iterable = ns.caches
  iterable.x = x
  iterable.b = y

  for item = iterable end
  return x
end

function Gridap.Algebra.solve!(::SSORIterativeSolverType,
                               x::PVector,
                               ns::IterativeLinearSolverNS,
                               y::PVector)
  iterables = ns.caches
  map(iterables,own_values(x),own_values(y)) do iterable, xi, yi
    iterable.x .= xi
    iterable.b .= yi
    for item = iterable end
    xi .= iterable.x
    yi .= iterable.b
  end
  consistent!(x) |> fetch
  return x
end
