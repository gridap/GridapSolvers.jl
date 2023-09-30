

# MINRES Solver
struct MINRESSolver <: Gridap.Algebra.LinearSolver
  m   ::Int
  Pl  ::Gridap.Algebra.LinearSolver
  atol::Float64
  rtol::Float64
  verbose::Bool
end

function MINRESSolver(m,Pl;atol=1e-12,rtol=1.e-6,verbose=false)
  return MINRESSolver(m,Pl,atol,rtol,verbose)
end

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

function get_MINRES_caches(m,A)
  w = allocate_col_vector(A)
  V = [allocate_col_vector(A) for i in 1:3]
  Z = [allocate_col_vector(A) for i in 1:3]

  H = zeros(m+1,m)  # Hessenberg matrix
  g = zeros(m+1)    # Residual vector
  c = zeros(m)      # Gibens rotation cosines
  s = zeros(m)      # Gibens rotation sines
  return (w,V,Z,H,g,c,s)
end

function Gridap.Algebra.numerical_setup(ss::MINRESSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pl_ns  = numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_MINRES_caches(solver.m,A)
  return MINRESNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::MINRESNumericalSetup, A::AbstractMatrix)
  numerical_setup!(ns.Pl_ns,A)
  ns.A = A
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::MINRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  m, atol, rtol, verbose = solver.m, solver.atol, solver.rtol, solver.verbose
  w, V, Z, H, g, c, s = caches
  verbose && println(" > Starting MINRES solver: ")

  # Initial residual
  mul!(w,A,x); w .= b .- w

  β    = norm(w); β0 = β
  converged = (β < atol || β < rtol*β0)
  iter = 0
  while !converged
    verbose && println("   > Iteration ", iter," - Residual: ", β)
    fill!(H,0.0)
    
    # Arnoldi process
    fill!(g,0.0); g[1] = β
    V[1] .= w ./ β
    j = 1
    
    # Arnoldi orthogonalization by Modified Gram-Schmidt
    solve!(Z[j],Pl,V[j])
    mul!(w,A,Z[j])
    for i in 1:j
      H[i,j] = dot(w,V[i])
      w .= w .- H[i,j] .* V[i]
    end
    H[j+1,j] = norm(w)
    V[j+1] = w ./ H[j+1,j]

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

    β  = abs(g[j+1]); converged = (β < atol || β < rtol*β0)

    # Solve least squares problem Hy = g by backward substitution
    for i in j:-1:1
      g[i] = (g[i] - dot(H[i,i+1:j],g[i+1:j])) / H[i,i]
    end

    # Update solution & residual
    for i in 1:j
      x .+= g[i] .* Z[i]
    end
    mul!(w,A,x); w .= b .- w

    iter += 1
  end
  verbose && println("   > Num Iter: ", iter," - Final residual: ", β)
  verbose && println("   Exiting MINRES solver.")

  return x
end




