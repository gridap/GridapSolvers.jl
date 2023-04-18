
struct SymGaussSeidelSmoother <: Gridap.Algebra.LinearSolver
  num_iters::Int
end

struct SymGaussSeidelSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: SymGaussSeidelSmoother
end

function Gridap.Algebra.symbolic_setup(s::SymGaussSeidelSmoother,A::AbstractMatrix)
  SymGaussSeidelSymbolicSetup(s)
end

# Numerical setup

struct SymGaussSeidelNumericalSetup{A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver :: SymGaussSeidelSmoother
  mat    :: A
  L      :: B
  U      :: C
  caches :: D
end

function _gs_get_caches(A::AbstractMatrix)
  dx  = allocate_col_vector(A)
  Adx = allocate_row_vector(A)
  return dx, Adx
end

function _gs_decompose_matrix(A::AbstractMatrix)
  D  = IterativeSolvers.DiagonalIndices(A)
  L  = IterativeSolvers.FastLowerTriangular(A, D)
  U  = IterativeSolvers.FastUpperTriangular(A, D)
  return L,U
end

function Gridap.Algebra.numerical_setup(ss::SymGaussSeidelSymbolicSetup,A::AbstractMatrix)
  L, U   = _gs_decompose_matrix(A)
  caches = _gs_get_caches(A)
  return SymGaussSeidelNumericalSetup(ss.solver,A,L,U,caches)
end

function Gridap.Algebra.numerical_setup(ss::SymGaussSeidelSymbolicSetup,A::PSparseMatrix)
  L,U = map_parts(A.owned_owned_values) do A
    # TODO: Unfortunately, we need to convert to CSC because the type is hardcoded in IterativeSolvers
    _gs_decompose_matrix(SparseMatrixCSC(A))
  end
  caches = _gs_get_caches(A)
  return SymGaussSeidelNumericalSetup(ss.solver,A,L,U,caches)
end

# Forward/backward substitution

function forward_sub!(L,dx::AbstractArray)
  IterativeSolvers.forward_sub!(L, dx)
end

function forward_sub!(L,dx::PVector)
  map_parts(L,dx.owned_values) do L, dx
    IterativeSolvers.forward_sub!(L, dx)
  end
end

function backward_sub!(U,dx::AbstractArray)
  IterativeSolvers.backward_sub!(U, dx)
end

function backward_sub!(U,dx::PVector)
  map_parts(U,dx.owned_values) do U, dx
    IterativeSolvers.backward_sub!(U, dx)
  end
end

# Solve

function Gridap.Algebra.solve!(x::AbstractVector, ns::SymGaussSeidelNumericalSetup, r::AbstractVector)
  A, L, U, caches = ns.mat, ns.L, ns.U, ns.caches
  dx, Adx = caches
  niter   = ns.solver.num_iters

  iter = 1
  while iter <= niter
    # Forward pass
    copy!(dx,r)
    forward_sub!(L, dx)
    x  .= x .+ dx
    mul!(Adx, A, dx)
    r  .= r .- Adx

    # Backward pass
    copy!(dx,r)
    backward_sub!(U, dx)
    x  .= x .+ dx
    mul!(Adx, A, dx)
    r  .= r .- Adx

    iter += 1
  end

  return x
end

function LinearAlgebra.ldiv!(x::AbstractVector, ns::SymGaussSeidelNumericalSetup, b::AbstractVector)
  fill!(x,0.0)
  aux = copy(b)
  solve!(x,ns,aux)
end
