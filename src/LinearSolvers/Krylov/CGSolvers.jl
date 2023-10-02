
struct CGSolver <: Gridap.Algebra.LinearSolver
  Pl       :: Gridap.Algebra.LinearSolver
  info     :: SolverTolerances{Float64}
  flexible :: Bool
  verbose  :: Bool
end

function CGSolver(Pl;maxiter=1000,atol=1e-12,rtol=1.e-6,flexible=false,verbose=false)
  tols = SolverTolerances{Float64}(maxiter=maxiter,atol=atol,rtol=rtol)
  return CGSolver(Pl,tols,flexible,verbose)
end

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

function get_solver_caches(solver::CGSolver,A)
  w = allocate_col_vector(A)
  p = allocate_col_vector(A)
  z = allocate_col_vector(A)
  r = allocate_col_vector(A)
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
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::CGNumericalSetup,b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  maxiter, atol, rtol = solver.maxiter, solver.atol, solver.rtol
  flexible, verbose = solver.flexible, solver.verbose
  w,p,z,r = caches
  verbose && println(" > Starting CG solver: ")

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,0.0); γ = 1.0

  res  = norm(r); res_0 = res
  iter = 0; converged = false
  while !converged && (iter < maxiter)
    verbose && println("   > Iteration ", iter," - Residual: ", res)

    if !flexible # β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
      solve!(z, Pl, r)
      β = γ; γ = dot(z, r); β = γ / β
    else         # β = (zₖ₊₁ ⋅ (rₖ₊₁-rₖ))/(zₖ ⋅ rₖ)
      β = γ; γ = dot(z, r)
      solve!(z, Pl, r)
      γ = dot(z, r) - γ; β = γ / β
    end
    p .= z .+ β .* p

    # w = A⋅p
    mul!(w,A,p)
    α = γ / dot(p, w)

    # Update solution and residual
    x .+= α .* p
    r .-= α .* w

    res = norm(r)
    converged = (res < atol || res < rtol*res_0)
    iter += 1
  end
  verbose && println("   > Num Iter: ", iter," - Final residual: ", res)
  verbose && println("   Exiting CG solver.")

  return x
end
