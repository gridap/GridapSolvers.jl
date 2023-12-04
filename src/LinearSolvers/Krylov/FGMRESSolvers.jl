
# FGMRES Solver
struct FGMRESSolver <: Gridap.Algebra.LinearSolver
  m         :: Int
  Pr        :: Gridap.Algebra.LinearSolver
  Pl        :: Union{Gridap.Algebra.LinearSolver,Nothing}
  outer_log :: ConvergenceLog{Float64}
  inner_log :: ConvergenceLog{Float64}
end

function FGMRESSolver(m,Pr;Pl=nothing,maxiter=100,atol=1e-12,rtol=1.e-6,verbose=false,name="FGMRES")
  outer_tols = SolverTolerances{Float64}(maxiter=maxiter,atol=atol,rtol=rtol)
  outer_log  = ConvergenceLog(name,outer_tols,verbose=verbose)
  inner_tols = SolverTolerances{Float64}(maxiter=m,atol=atol,rtol=rtol)
  inner_log  = ConvergenceLog("$(name)_inner",inner_tols,verbose=verbose,nested=true)
  return FGMRESSolver(m,Pr,Pl,outer_log,inner_log)
end

AbstractTrees.children(s::FGMRESSolver) = [s.Pr,s.Pl]

struct FGMRESSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::FGMRESSolver, A::AbstractMatrix)
  return FGMRESSymbolicSetup(solver)
end

mutable struct FGMRESNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
  A
  Pr_ns
  Pl_ns
  caches
end

function get_solver_caches(solver::FGMRESSolver,A)
  m = solver.m

  V  = [allocate_in_domain(A) for i in 1:m+1]
  Z  = [allocate_in_domain(A) for i in 1:m]
  zl = allocate_in_domain(A)

  H = zeros(m+1,m)  # Hessenberg matrix
  g = zeros(m+1)    # Residual vector
  c = zeros(m)      # Gibens rotation cosines
  s = zeros(m)      # Gibens rotation sines
  return (V,Z,zl,H,g,c,s)
end

function Gridap.Algebra.numerical_setup(ss::FGMRESSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pr_ns  = numerical_setup(symbolic_setup(solver.Pr,A),A)
  Pl_ns  = isa(solver.Pl,Nothing) ? nothing : numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_solver_caches(solver,A)
  return FGMRESNumericalSetup(solver,A,Pr_ns,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::FGMRESNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Pr_ns,A)
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A)
  end
  ns.A = A
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::FGMRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, Pr, caches = ns.solver, ns.A, ns.Pl_ns, ns.Pr_ns, ns.caches
  log, ilog = solver.outer_log, solver.inner_log
  V, Z, zl, H, g, c, s = caches

  # Initial residual
  krylov_residual!(V[1],x,A,b,Pl,zl)
  β    = norm(V[1])
  done = init!(log,β)
  while !done
    # Arnoldi process
    j = 1
    V[1] ./= β
    fill!(H,0.0)
    fill!(g,0.0); g[1] = β
    idone = init!(ilog,β)
    while !idone
      # Arnoldi orthogonalization by Modified Gram-Schmidt
      krylov_mul!(V[j+1],A,V[j],Pr,Pl,Z[j],zl)
      for i in 1:j
        H[i,j] = dot(V[j+1],V[i])
        V[j+1] .= V[j+1] .- H[i,j] .* V[i]
      end
      H[j+1,j] = norm(V[j+1])
      V[j+1] ./= H[j+1,j]

      # Update QR
      for i in 1:j-1
        γ = c[i]*H[i,j] + s[i]*H[i+1,j]
        H[i+1,j] = -s[i]*H[i,j] + c[i]*H[i+1,j]
        H[i,j] = γ
      end

      # New Givens rotation, update QR and residual
      c[j], s[j], _ = LinearAlgebra.givensAlgorithm(H[j,j],H[j+1,j])
      H[j,j] = c[j]*H[j,j] + s[j]*H[j+1,j]; H[j+1,j] = 0.0
      g[j+1] = -s[j]*g[j]; g[j] = c[j]*g[j]

      β  = abs(g[j+1])
      j += 1
      idone = update!(ilog,β)
    end
    j = j-1

    # Solve least squares problem Hy = g by backward substitution
    for i in j:-1:1
      g[i] = (g[i] - dot(H[i,i+1:j],g[i+1:j])) / H[i,i]
    end

    # Update solution & residual
    for i in 1:j
      x .+= g[i] .* Z[i]
    end
    krylov_residual!(V[1],x,A,b,Pl,zl)
    done = update!(log,β)
  end

  finalize!(log,β)
  return x
end
