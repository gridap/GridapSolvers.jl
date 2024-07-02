"""
    struct CGSolver <: LinearSolver
      ...
    end

    CGSolver(Pl;maxiter=1000,atol=1e-12,rtol=1.e-6,flexible=false,verbose=0,name="CG")

  Left-Preconditioned Conjugate Gradient solver.
"""
struct CGSolver <: Gridap.Algebra.LinearSolver
  Pl       :: Gridap.Algebra.LinearSolver
  log      :: ConvergenceLog{Float64}
  flexible :: Bool
end

function CGSolver(Pl;maxiter=1000,atol=1e-12,rtol=1.e-6,flexible=false,verbose=0,name="CG")
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols;verbose=verbose)
  return CGSolver(Pl,log,flexible)
end

AbstractTrees.children(s::CGSolver) = [s.Pl]

struct CGSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::CGSolver, A::AbstractMatrix)
  return CGSymbolicSetup(solver)
end

mutable struct CGNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
  A
  Pl_ns
  caches
end

function get_solver_caches(solver::CGSolver,A::AbstractMatrix)
  w = allocate_in_domain(A); fill!(w,zero(eltype(w)))
  p = allocate_in_domain(A); fill!(p,zero(eltype(p)))
  z = allocate_in_domain(A); fill!(z,zero(eltype(z)))
  r = allocate_in_domain(A); fill!(r,zero(eltype(r)))
  return (w,p,z,r)
end

function Gridap.Algebra.numerical_setup(ss::CGSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pl_ns  = numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_solver_caches(solver,A)
  return CGNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::CGNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Pl_ns,A)
  ns.A = A
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::CGNumericalSetup, A::AbstractMatrix, x::AbstractVector)
  numerical_setup!(ns.Pl_ns,A,x)
  ns.A = A
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::CGNumericalSetup,b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  flexible, log = solver.flexible, solver.log
  w,p,z,r = caches

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,zero(eltype(p)))
  fill!(z,zero(eltype(z)))
  γ = one(eltype(p))

  res  = norm(r)
  done = init!(log,res)
  while !done

    if !flexible # β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
      solve!(z, Pl, r)
      β = γ; γ = dot(z, r); β = γ / β
    else         # β = (zₖ₊₁ ⋅ (rₖ₊₁-rₖ))/(zₖ ⋅ rₖ)
      δ = dot(z, r)
      solve!(z, Pl, r)
      β = γ; γ = dot(z, r); β = (γ-δ) / β
    end
    p .= z .+ β .* p

    # w = A⋅p
    mul!(w,A,p)
    α = γ / dot(p, w)

    # Update solution and residual
    x .+= α .* p
    r .-= α .* w

    res  = norm(r)
    done = update!(log,res)
  end

  finalize!(log,res)
  return x
end
