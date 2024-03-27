
"""
    struct FGMRESSolver <: LinearSolver 
      ...
    end

    FGMRESSolver(m,Pr;Pl=nothing,restart=false,m_add=1,maxiter=100,atol=1e-12,rtol=1.e-6,verbose=false,name="FGMRES")

  Flexible GMRES solver, with right-preconditioner `Pr` and optional left-preconditioner `Pl`.

  The solver starts by allocating a basis of size `m`. Then: 

   - If `restart=true`, the basis size is fixed and restarted every `m` iterations.
   - If `restart=false`, the basis size is allowed to increase. When full, the solver 
     allocates `m_add` new basis vectors.
"""
struct FGMRESSolver <: Gridap.Algebra.LinearSolver
  m         :: Int
  restart   :: Bool
  m_add     :: Int
  Pr        :: Gridap.Algebra.LinearSolver
  Pl        :: Union{Gridap.Algebra.LinearSolver,Nothing}
  log :: ConvergenceLog{Float64}
end

function FGMRESSolver(m,Pr;Pl=nothing,restart=false,m_add=1,maxiter=100,atol=1e-12,rtol=1.e-6,verbose=false,name="FGMRES")
  tols = SolverTolerances{Float64}(maxiter=maxiter,atol=atol,rtol=rtol)
  log  = ConvergenceLog(name,tols,verbose=verbose)
  return FGMRESSolver(m,restart,m_add,Pr,Pl,log)
end

function restart(s::FGMRESSolver,k::Int)
  if s.restart && (k > s.m)
    print_message(s.log,"Restarting Krylov basis.")
    return true
  end
  return false
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

function krylov_cache_length(ns::FGMRESNumericalSetup)
  V, _, _, _, _, _, _ = ns.caches
  return length(V) - 1
end

function expand_krylov_caches!(ns::FGMRESNumericalSetup)
  V, Z, zl, H, g, c, s = ns.caches

  m = krylov_cache_length(ns)
  m_add = ns.solver.m_add
  m_new = m + m_add

  for _ in 1:m_add
    push!(V,allocate_in_domain(ns.A))
    push!(Z,allocate_in_domain(ns.A))
  end
  H_new = zeros(eltype(H),m_new+1,m_new); H_new[1:m+1,1:m] .= H
  g_new = zeros(eltype(g),m_new+1); g_new[1:m+1] .= g
  c_new = zeros(eltype(c),m_new); c_new[1:m] .= c
  s_new = zeros(eltype(s),m_new); s_new[1:m] .= s
  ns.caches = (V,Z,zl,H_new,g_new,c_new,s_new)
  return H_new,g_new,c_new,s_new
end

function Gridap.Algebra.numerical_setup(ss::FGMRESSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pr_ns  = numerical_setup(symbolic_setup(solver.Pr,A),A)
  Pl_ns  = !isnothing(solver.Pl) ? numerical_setup(symbolic_setup(solver.Pl,A),A) : nothing
  caches = get_solver_caches(solver,A)
  return FGMRESNumericalSetup(solver,A,Pr_ns,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup(ss::FGMRESSymbolicSetup, A::AbstractMatrix, x::AbstractVector)
  solver = ss.solver
  Pr_ns  = numerical_setup(symbolic_setup(solver.Pr,A,x),A,x)
  Pl_ns  = !isnothing(solver.Pl) ? numerical_setup(symbolic_setup(solver.Pl,A,x),A,x) : nothing
  caches = get_solver_caches(solver,A)
  return FGMRESNumericalSetup(solver,A,Pr_ns,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::FGMRESNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Pr_ns,A)
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A)
  end
  ns.A = A
  return ns
end

function Gridap.Algebra.numerical_setup!(ns::FGMRESNumericalSetup, A::AbstractMatrix, x::AbstractVector)
  numerical_setup!(ns.Pr_ns,A,x)
  if !isa(ns.Pl_ns,Nothing)
    numerical_setup!(ns.Pl_ns,A,x)
  end
  ns.A = A
  return ns
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::FGMRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, Pr, caches = ns.solver, ns.A, ns.Pl_ns, ns.Pr_ns, ns.caches
  V, Z, zl, H, g, c, s = caches
  m   = krylov_cache_length(ns)
  log = solver.log

  fill!(V[1],zero(eltype(V[1])))
  fill!(zl,zero(eltype(zl)))

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
    while !done && !restart(solver,j)
      # Expand Krylov basis if needed
      if j > m  
        H, g, c, s = expand_krylov_caches!(ns)
        m = krylov_cache_length(ns)
      end

      # Arnoldi orthogonalization by Modified Gram-Schmidt
      fill!(V[j+1],zero(eltype(V[j+1])))
      fill!(Z[j],zero(eltype(Z[j])))
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
      done = update!(log,β)
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
  end

  finalize!(log,β)
  return x
end
