"""
    struct MINRESSolver <: LinearSolver 
      ...
    end

    MINRESSolver(m;Pr=nothing,Pl=nothing,maxiter=100,atol=1e-12,rtol=1.e-6,verbose=false,name="MINRES")

  MINRES solver, with optional right and left preconditioners `Pr` and `Pl`.
"""
struct MINRESSolver <: Gridap.Algebra.LinearSolver
  Pr  :: Union{Gridap.Algebra.LinearSolver,Nothing}
  Pl  :: Union{Gridap.Algebra.LinearSolver,Nothing}
  log :: ConvergenceLog{Float64}
end

function MINRESSolver(;Pr=nothing,Pl=nothing,maxiter=1000,atol=1e-12,rtol=1.e-6,verbose=false,name="MINRES")
  tols = SolverTolerances{Float64}(maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols,verbose=verbose)
  return MINRESSolver(Pr,Pl,log)
end

AbstractTrees.children(s::MINRESSolver) = [s.Pr,s.Pl]

struct MINRESSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::MINRESSolver, A::AbstractMatrix)
  return MINRESSymbolicSetup(solver)
end

mutable struct MINRESNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
  A
  Pr_ns
  Pl_ns
  caches
end

function get_solver_caches(solver::MINRESSolver,A)
  Pl, Pr = solver.Pl, solver.Pr

  V  = [allocate_in_domain(A) for i in 1:3]
  W  = [allocate_in_domain(A) for i in 1:3]
  zr = !isa(Pr,Nothing) ? allocate_in_domain(A) : nothing
  zl = !isa(Pl,Nothing) ? allocate_in_domain(A) : nothing

  H = zeros(4) # Hessenberg matrix
  g = zeros(2) # Residual vector
  c = zeros(2) # Gibens rotation cosines
  s = zeros(2) # Gibens rotation sines
  return (V,W,zr,zl,H,g,c,s)
end

function Gridap.Algebra.numerical_setup(ss::MINRESSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pr_ns  = isa(solver.Pr,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pr,A),A)
  Pl_ns  = isa(solver.Pl,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_solver_caches(solver,A)
  return MINRESNumericalSetup(solver,A,Pr_ns,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup(ss::MINRESSymbolicSetup, A::AbstractMatrix, x::AbstractVector)
  solver = ss.solver
  Pr_ns  = isa(solver.Pr,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pr,A,x),A,x)
  Pl_ns  = isa(solver.Pl,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pl,A,x),A,x)
  caches = get_solver_caches(solver,A)
  return MINRESNumericalSetup(solver,A,Pr_ns,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::MINRESNumericalSetup, A::AbstractMatrix)
  if !isa(ns.Pr_ns,Nothing)
    numerical_setup!(ns.Pr_ns,A)
  end
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A)
  end
  ns.A = A
end

function Gridap.Algebra.numerical_setup!(ns::MINRESNumericalSetup, A::AbstractMatrix, x::AbstractVector)
  if !isa(ns.Pr_ns,Nothing)
    numerical_setup!(ns.Pr_ns,A,x)
  end
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A,x)
  end
  ns.A = A
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::MINRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, Pr, caches = ns.solver, ns.A, ns.Pl_ns, ns.Pr_ns, ns.caches
  V, W, zr, zl, H, g, c, s = caches
  log = solver.log

  Vjm1, Vj, Vjp1 = V
  Wjm1, Wj, Wjp1 = W

  fill!(zr,0.0); fill!(zl,0.0)
  fill!(Vjm1,0.0); fill!(Vjp1,0.0); copy!(Vj,b)
  fill!(Wjm1,0.0); fill!(Wjp1,0.0); fill!(Wj,0.0)
  fill!(H,0.0); fill!(c,1.0); fill!(s,0.0); fill!(g,0.0)

  krylov_residual!(Vj,x,A,b,Pl,zl)
  β    = norm(Vj); Vj ./= β; g[1] = β
  done = init!(log,β)
  while !done
    # Lanczos process
    krylov_mul!(Vjp1,A,Vj,Pr,Pl,zr,zl)
    H[3] = dot(Vjp1,Vj)
    Vjp1 .= Vjp1 .- H[3] .* Vj .- H[2] .* Vjm1
    H[4] = norm(Vjp1)
    Vjp1 ./= H[4]

    # Update QR
    H[1] = s[1]*H[2]; H[2] = c[1]*H[2]
    γ = c[2]*H[2] + s[2]*H[3]; H[3] = -s[2]*H[2] + c[2]*H[3]; H[2] = γ

    # New Givens rotation, update QR and residual
    c[1], s[1] = c[2], s[2]
    c[2], s[2], H[3] = LinearAlgebra.givensAlgorithm(H[3],H[4])
    g[2] = -s[2]*g[1]; g[1] = c[2]*g[1]

    # Update solution
    Wjp1 .= Vj .- H[2] .* Wj .- H[1] .* Wjm1
    Wjp1 ./= H[3]
    if isa(Pr,Nothing)
      x .+= g[1] .* Wjp1
    else
      solve!(zr,Pr,Wjp1)
      x .+= g[1] .* zr
    end

    β  = abs(g[2])
    Vjm1, Vj, Vjp1 = Vj, Vjp1, Vjm1
    Wjm1, Wj, Wjp1 = Wj, Wjp1, Wjm1
    g[1] = g[2]; H[2] = H[4];
    done = update!(log,β)
  end
  
  finalize!(log,β)
  return x
end