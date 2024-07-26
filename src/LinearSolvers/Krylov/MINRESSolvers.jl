"""
    struct MINRESSolver <: LinearSolver 
      ...
    end

    MINRESSolver(m;Pl=nothing,maxiter=100,atol=1e-12,rtol=1.e-6,verbose=false,name="MINRES")

  MINRES solver, with optional left preconditioners `Pl`. The preconditioner must be 
  symmetric and positive definite.
"""
struct MINRESSolver <: Gridap.Algebra.LinearSolver
  Pl  :: Union{Gridap.Algebra.LinearSolver,Nothing}
  log :: ConvergenceLog{Float64}
end

function MINRESSolver(;Pl=nothing,maxiter=1000,atol=1e-12,rtol=1.e-6,verbose=false,name="MINRES")
  tols = SolverTolerances{Float64}(maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols,verbose=verbose)
  return MINRESSolver(Pl,log)
end

AbstractTrees.children(s::MINRESSolver) = [s.Pl]

struct MINRESSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::MINRESSolver, A::AbstractMatrix)
  return MINRESSymbolicSetup(solver)
end

mutable struct MINRESNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
  A
  Pl_ns
  caches
end

function get_solver_caches(solver::MINRESSolver,A::AbstractMatrix)
  V = Tuple([allocate_in_domain(A) for i in 1:3])
  W = Tuple([allocate_in_domain(A) for i in 1:3])
  Z = Tuple([allocate_in_domain(A) for i in 1:3])
  return (V,W,Z)
end

function Gridap.Algebra.numerical_setup(ss::MINRESSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pl_ns  = isa(solver.Pl,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_solver_caches(solver,A)
  return MINRESNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup(ss::MINRESSymbolicSetup, A::AbstractMatrix, x::AbstractVector)
  solver = ss.solver
  Pl_ns  = isa(solver.Pl,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pl,A,x),A,x)
  caches = get_solver_caches(solver,A)
  return MINRESNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::MINRESNumericalSetup, A::AbstractMatrix)
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A)
  end
  ns.A = A
end

function Gridap.Algebra.numerical_setup!(ns::MINRESNumericalSetup, A::AbstractMatrix, x::AbstractVector)
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A,x)
  end
  ns.A = A
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::MINRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  Vs, Ws, Zs = caches
  log = solver.log

  Vnew, V, Vold = Vs
  Wnew, W, Wold = Ws
  Znew, Z, Zold = Zs

  T = eltype(A)
  fill!(W,zero(T))
  fill!(Wold,zero(T))
  fill!(Vold,zero(T))
  fill!(Zold,zero(T))

  mul!(Vnew,A,x)
  Vnew .= b .- Vnew
  fill!(Znew,zero(T))
  !isnothing(Pl) ? solve!(Znew,Pl,Vnew) : copy!(Znew,Vnew)

  β_r = norm(Znew)
  β_p = dot(Znew,Vnew)
  @check β_p > zero(T)

  γnew, γ, γold = zero(T), sqrt(β_p), one(T)
  cnew, c, cold = zero(T), one(T), one(T)
  snew, s, sold = zero(T), zero(T), zero(T)

  V .= Vnew ./ γ
  Z .= Znew ./ γ
  
  η = γ
  done = init!(log,β_r)
  while !done
    # Lanczos process
    mul!(Vnew,A,Z)
    !isnothing(Pl) ? solve!(Znew,Pl,Vnew) : copy!(Znew,Vnew)
    δ = dot(Vnew,Z)
    Vnew .= Vnew .- δ .* V .- γ .* Vold
    Znew .= Znew .- δ .* Z .- γ .* Zold
    β_p = dot(Znew,Vnew)
    γnew = sqrt(β_p)

    Vnew .= Vnew ./ γnew
    Znew .= Znew ./ γnew

    # Update QR
    α0 = c*δ - cold*s*γ
    cnew, snew, α1 = LinearAlgebra.givensAlgorithm(α0,γnew)
    α2 = s*δ + cold*c*γ
    α3 = sold*γ

    # Update solution
    Wnew .= (Z .- α2 .* W .- α3 .* Wold) ./ α1
    x .= x .+ (cnew*η) .* Wnew
    η = - snew * η

    # Update residual
    β_r = abs(snew) * β_r

    # Swap variables
    swap3(xnew,x,xold) = xold, xnew, x
    Vnew, V, Vold = swap3(Vnew, V, Vold)
    Wnew, W, Wold = swap3(Wnew, W, Wold)
    Znew, Z, Zold = swap3(Znew, Z, Zold)
    γnew, γ, γold = swap3(γnew, γ, γold)
    cnew, c, cold = swap3(cnew, c, cold)
    snew, s, sold = swap3(snew, s, sold)

    done = update!(log,β_r)
  end
  
  finalize!(log,β_r)
  return x
end