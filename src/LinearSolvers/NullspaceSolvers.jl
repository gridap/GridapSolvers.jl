
"""
    struct NullspaceSolver{A,B}
      solver :: A <: LinearSolver
      kernel :: B <: Union{AbstractMatrix, Vector{<:AbstractVector}}
      constrain_matrix :: Bool = true
    end

Solver that computes the solution of a linear system `A⋅x = b` with a kernel constraint,
i.e the returned solution is orthogonal to the provided kernel.

We assume the kernel is provided as a matrix `K` of dimensions `(k,n)` with `n` the dimension
of the original system and `k` the number of kernel vectors.

Two modes of operation are supported:

- If `constrain_matrix` is `true`, the solver will explicitly constrain the system matrix. 
  I.e we consider the augmented system `Â⋅x̂ = b̂` where
  -  `Ak = [A, K'; K, 0]`
  -  `x̂ = [x; λ]`
  -  `b̂ = [b; 0]`
  This is often the only option for direct solvers, which require the system matrix to be
  invertible. This is only supported in serial (its performance bottleneck in parallel).

- If `constrain_matrix` is `false`, the solver preserve the original system and simply 
  project the initial guess and the solution onto the orthogonal complement of the kernel. 
  This option is more suitable for iterative solvers, which usually do not require the 
  system matrix to be invertible (e.g. GMRES, BiCGStab, etc).
"""
struct NullspaceSolver{A,B} <: Gridap.Algebra.LinearSolver
  solver    :: A
  nullspace :: B
  constrain_matrix :: Bool

  function NullspaceSolver(
    solver::LinearSolver,
    nullspace::NullSpace; 
    constrain_matrix::Bool = true
  )
    A, B = typeof(solver), typeof(nullspace)
    new{A,B}(solver, nullspace, constrain_matrix)
  end
end

struct NullspaceSolverSS{A} <: Algebra.SymbolicSetup
  solver :: A
end

function Gridap.Algebra.symbolic_setup(solver::NullspaceSolver,A::AbstractMatrix)
  return NullspaceSolverSS(solver)
end

struct NullspaceSolverNS{S}
  solver
  ns
  caches
end

function Algebra.numerical_setup(ss::NullspaceSolverSS, A::AbstractMatrix)
  solver = ss.solver
  N = solver.nullspace
  nK, nV = size(N)
  @assert size(A,1) == size(A,2) == nV
  if solver.constrain_matrix
    K = SolverInterfaces.matrix_representation(N)
    mat = [A K; K' zeros(nK,nK)] # TODO: Explore reusing storage for A
  else
    SolverInterfaces.make_orthonormal!(N)
    mat = A
  end
  S = ifelse(solver.constrain_matrix, :constrained, :projected)
  ns = numerical_setup(symbolic_setup(solver.solver, mat), mat)
  caches = (allocate_in_domain(mat), allocate_in_domain(mat))
  return NullspaceSolverNS{S}(solver, ns, caches)
end

function Algebra.numerical_setup!(ns::NullspaceSolverNS, A::AbstractMatrix)
  solver = ns.solver
  N = solver.nullspace
  nK, nV = size(N)
  @assert size(A,1) == size(A,2) == nV
  if solver.constrain_matrix
    K = SolverInterfaces.matrix_representation(N)
    mat = [A K; K' zeros(nK,nK)] # TODO: Explore reusing storage for A
  else
    mat = A
  end
  numerical_setup!(ns.ns, mat)
  return ns
end

function Algebra.solve!(x::AbstractVector, ns::NullspaceSolverNS{:constrained}, b::AbstractVector)
  solver = ns.solver
  A_ns, caches = ns.ns, ns.caches
  N = solver.nullspace
  w1, w2 = caches
  nK, nV = size(N)
  @assert length(x) == nV
  @assert length(b) == nV
  w1[1:nV] .= x
  w1[nV+1:nV+nK] .= 0.0
  w2[1:nV] .= b
  w2[nV+1:nV+nK] .= 0.0
  solve!(w1, A_ns, w2)
  x .= w1[1:nV]
  return x
end

function Algebra.solve!(x::AbstractVector, ns::NullspaceSolverNS{:projected}, b::AbstractVector)
  solver = ns.solver
  A_ns, caches = ns.ns, ns.caches
  N = solver.nullspace
  w1, w2 = caches
  
  w1, α = SolverInterfaces.project!(w1, N, x)
  x .-= w1
  solve!(x, A_ns, b)

  return x
end
