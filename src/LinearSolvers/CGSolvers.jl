
struct CGSolver <: Gridap.Algebra.LinearSolver
  Pl      :: Gridap.Algebra.LinearSolver
  maxiter :: Int64
  atol    :: Float64
  rtol    :: Float64
  variant :: Symbol
  verbose :: Bool
end

function CGSolver(Pl;maxiter=10000,atol=1e-12,rtol=1.e-6,flexible=false,verbose=false)
  variant = flexible ? :flexible : :standard
  return CGSolver(Pl,maxiter,atol,rtol,variant,verbose)
end

struct CGSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::CGSolver, A::AbstractMatrix)
  return CGSymbolicSetup(solver)
end

mutable struct CGNumericalSetup{T} <: Gridap.Algebra.NumericalSetup
  solver
  A
  Pl_ns
  caches
end

function get_cg_caches(A)
  w = allocate_col_vector(A)
  p = allocate_col_vector(A)
  z = allocate_col_vector(A)
  r = allocate_col_vector(A)
  return (w,p,z,r)
end

function Gridap.Algebra.numerical_setup(ss::CGSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pl_ns  = numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_cg_caches(A)
  return CGNumericalSetup{solver.variant}(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::CGNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Pl_ns,A)
  ns.A = A
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::CGNumericalSetup{:standard},b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  maxiter, atol, rtol, verbose = solver.maxiter, solver.atol, solver.rtol, solver.verbose
  w,p,z,r = caches
  verbose && println(" > Starting CG solver: ")

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,0.0); γ = 1.0

  res  = norm(r); res_0 = res
  iter = 0; converged = false
  while !converged && (iter < maxiter)
    verbose && println("   > Iteration ", iter," - Residual: ", res)

    # Apply left preconditioner
    solve!(z, Pl, r)

    # p := z + β⋅p , β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
    β = γ; γ = dot(z, r); β = γ / β
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

function Gridap.Algebra.solve!(x::AbstractVector,ns::CGNumericalSetup{:flexible},b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  maxiter, atol, rtol, verbose = solver.maxiter, solver.atol, solver.rtol, solver.verbose
  w,p,z,r = caches
  verbose && println(" > Starting CG solver: ")

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,0.0); γ = 1.0

  res  = norm(r); res_0 = res
  iter = 0; converged = false
  while !converged && (iter < maxiter)
    verbose && println("   > Iteration ", iter," - Residual: ", res)

    # p := z + β⋅p , β = (zₖ₊₁ ⋅ (rₖ₊₁-rₖ))/(zₖ ⋅ rₖ)
    β = γ; γ = dot(z, r)
    solve!(z, Pl, r)
    γ = dot(z, r) - γ; β = γ / β
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
