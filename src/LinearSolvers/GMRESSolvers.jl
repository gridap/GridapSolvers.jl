
# Orthogonalization




# GMRES Solver
struct GMRESSolver <: Gridap.Algebra.LinearSolver
  m  ::Int
  Pl
  tol::Float64
end

struct GMRESSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver
end

function Gridap.Algebra.symbolic_setup(solver::GMRESSolver, A::AbstractMatrix)
  return GMRESSymbolicSetup(solver)
end

struct GMRESNumericalSetup <: Gridap.Algebra.NumericalSetup
  solver
  A
  Pl_ns
  caches
end

function get_gmres_caches(m,A)
  w = allocate_col_vector(A)
  V = [allocate_col_vector(A) for i in 1:m+1]
  Z = [allocate_col_vector(A) for i in 1:m]

  H = zeros(m+1,m)  # Hessenberg matrix
  g = zeros(m+1)    # Residual vector
  c = zeros(m)      # Gibens rotation cosines
  s = zeros(m)      # Gibens rotation sines
  return (w,V,Z,H,g,c,s)
end

function Gridap.Algebra.numerical_setup(ss::GMRESSymbolicSetup, A::AbstractMatrix)
  solver = ss.solver
  Pl_ns  = numerical_setup(symbolic_setup(solver.Pl,A),A)
  caches = get_gmres_caches(solver.m,A)
  return GMRESNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.solve!(x::AbstractVector,ns::GMRESNumericalSetup,b::AbstractVector)
  solver, A, Pl, caches = ns.solver, ns.A, ns.Pl_ns, ns.caches
  m, tol = solver.m, solver.tol
  w, V, Z, H, g, c, s = caches
  println(" > Starting GMRES solve: ")

  # Initial residual
  mul!(w,A,x); w .= b .- w

  β    = norm(w)
  iter = 0
  while (β > tol)
    println("   > Iteration ", iter," - Residual: ", β)
    fill!(H,0.0)
    
    # Arnoldi process
    fill!(g,0.0); g[1] = β
    V[1] .= w ./ β
    for j in 1:m
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

      β = abs(g[j+1])
    end

    # Solve least squares problem Hy = g by backward substitution
    for i in m:-1:1
      g[i] = (g[i] - dot(H[i,i+1:m],g[i+1:m])) / H[i,i]
    end

    # Update solution & residual
    for i in 1:m
      x .+= g[i] .* Z[i]
    end
    mul!(w,A,x); w .= b .- w

    iter += 1
  end
  println("   > Iteration ", iter," - Residual: ", β)

  return x
end
