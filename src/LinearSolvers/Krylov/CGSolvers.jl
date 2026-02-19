"""
    struct CGSolver <: LinearSolver
      ...
    end

    CGSolver(Pl;maxiter=1000,atol=1e-12,rtol=1.e-6,flexible=false,verbose=0,name="CG")

  Left-Preconditioned Conjugate Gradient solver.
"""
struct CGSolver{TL<:NothingOrSolver,TD} <: Gridap.Algebra.LinearSolver
  Pl       :: TL
  log      :: ConvergenceLog{Float64}
  diag     :: TD
  flexible :: Bool
end

CGSolver(;Pl=nothing,kwargs...) = CGSolver(Pl;kwargs...)

function CGSolver(Pl;maxiter=1000,atol=1e-12,rtol=1.e-6,diagnostic=nothing,flexible=false,verbose=0,name="CG")
  tols = SolverTolerances{Float64}(;maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols;verbose=verbose)
  return CGSolver(Pl,log,diagnostic,flexible)
end

AbstractTrees.children(s::CGSolver) = [s.Pl]

struct CGSymbolicSetup{A} <: Gridap.Algebra.SymbolicSetup
  solver::A
end

function Gridap.Algebra.symbolic_setup(solver::CGSolver, A::AbstractMatrix)
  return CGSymbolicSetup(solver)
end

mutable struct CGNumericalSetup{A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver::A
  mat::B
  Pl_ns::C
  caches::D
end

function get_solver_caches(::CGSolver,A::AbstractMatrix)
  w = allocate_in_domain(A); fill!(w,zero(eltype(w)))
  p = allocate_in_domain(A); fill!(p,zero(eltype(p)))
  z = allocate_in_domain(A); fill!(z,zero(eltype(z)))
  r = allocate_in_domain(A); fill!(r,zero(eltype(r)))
  return (w,p,z,r)
end

function Algebra.numerical_setup(ss::CGSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pl_ns  = !isnothing(solver.Pl) ? numerical_setup(symbolic_setup(solver.Pl,A),A) : nothing
  caches = get_solver_caches(solver,A)
  return CGNumericalSetup(solver,A,Pl_ns,caches)
end

function Algebra.numerical_setup!(ns::CGNumericalSetup, A::AbstractMatrix)
  if !isnothing(ns.Pl_ns)
    numerical_setup!(ns.Pl_ns,A)
  end
  ns.mat = A
  return ns
end

function Algebra.numerical_setup!(ns::CGNumericalSetup, A::AbstractMatrix, x::AbstractVector)
  if !isnothing(ns.Pl_ns)
    numerical_setup!(ns.Pl_ns,A,x)
  end
  ns.mat = A
  return ns
end

function Algebra.solve!(x::AbstractVector,ns::CGNumericalSetup,b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.mat, ns.Pl_ns, ns.caches
  flexible, log = solver.flexible, solver.log
  w,p,z,r = caches

  # Initial residual
  mul!(w,A,x); r .= b .- w
  fill!(p,zero(eltype(p)))
  fill!(z,zero(eltype(z)))
  γ = one(eltype(p))
  α_last = one(eltype(p))

  res  = norm(r)
  done = init!(log,res)
  cg_reset_diagnostic!(solver.diag)
  while !done

    if isnothing(Pl)
      z .= r
      β = γ; γ = dot(r, r); β = γ / β
    elseif !flexible # β = (zₖ₊₁ ⋅ rₖ₊₁)/(zₖ ⋅ rₖ)
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

    cg_update_diagnostic!(solver.diag, α, β, α_last)
    α_last = α
  end

  finalize!(log,res)
  return x
end

@inline cg_reset_diagnostic!(::Nothing, args...) = nothing

@inline cg_reset_diagnostic!(x::LanczosDiagnostic) = reset!(x)

@inline cg_update_diagnostic!(::Nothing, args...) = nothing

@inline function cg_update_diagnostic!(x::LanczosDiagnostic, αk::Real, βk::Real, αkm1::Real)
  k = x.k[]
  if iszero(k)
    δ = 1.0 / αk
    γ = 0.0
  else
    δ = (1.0 / αk) + (βk / αkm1)
    γ = sqrt(βk) / αk
  end
  return update!(x,δ,γ)
end
