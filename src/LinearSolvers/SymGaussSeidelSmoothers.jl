
# Extensions of IterativeSolvers.jl to support parallel matrices
struct DiagonalIndices{Tv,Ti,A,B}
  mat  :: A
  diag :: B
  last :: B
  function DiagonalIndices(mat ::AbstractSparseMatrix{Tv,Ti},
                           diag::AbstractVector{Ti},
                           last::AbstractVector{Ti}) where {Tv,Ti}
    A = typeof(mat)
    B = typeof(diag)
    @check typeof(last) == B
    new{Tv,Ti,A,B}(mat,diag,last)
  end
end

function DiagonalIndices(A::SparseMatrixCSR{Tv,Ti},row_range) where {Tv,Ti}
  @notimplemented
end

function DiagonalIndices(A::SparseMatrixCSC{Tv,Ti},col_range) where {Tv,Ti}
  n    = length(col_range)
  diag = Vector{Ti}(undef, n)
  last = Vector{Ti}(undef, n)

  for col in col_range
    # Diagonal index
    r1 = Int(A.colptr[col])
    r2 = Int(A.colptr[col + 1] - 1)
    r1 = searchsortedfirst(A.rowval, col, r1, r2, Base.Order.Forward)
    if r1 > r2 || A.rowval[r1] != col || iszero(A.nzval[r1])
        throw(LinearAlgebra.SingularException(col))
    end
    diag[col] = r1

    # Last owned index
    r1 = Int(A.colptr[col])
    r2 = Int(A.colptr[col + 1] - 1)
    r1 = searchsortedfirst(A.rowval, n+1, r1, r2, Base.Order.Forward) - 1
    last[col] = r1
  end
  return DiagonalIndices(A,diag,last)
end

struct LowerTriangular{Tv,Ti,A,B}
  mat  :: A
  diag :: DiagonalIndices{Tv,Ti,A,B}
end

struct UpperTriangular{Tv,Ti,A,B}
  mat  :: A
  diag :: DiagonalIndices{Tv,Ti,A,B}
end

function forward_sub!(L::LowerTriangular{Tv,Ti,<:SparseMatrixCSC},x::AbstractVector) where {Tv,Ti}
  A, diag, last = L.mat, L.diag.diag, L.diag.last
  n = length(diag)
  for col = 1 : n
    # Solve for diagonal element
    idx = diag[col]
    x[col] /= A.nzval[idx]

    # Substitute next values involving x[col]
    for i = idx + 1 : last[col]
      x[A.rowval[i]] -= A.nzval[i] * x[col]
    end
  end
  return x
end

function forward_sub!(L::AbstractArray{<:LowerTriangular},x::PVector)
  map(L,own_values(x)) do L, x
    forward_sub!(L, x)
  end
end

function backward_sub!(U::UpperTriangular{Tv,Ti,<:SparseMatrixCSC}, x::AbstractVector) where {Tv,Ti}
  A, diag = U.mat, U.diag.diag
  n = length(diag)
  for col = n : -1 : 1
    # Solve for diagonal element
    idx = diag[col]
    x[col] = x[col] / A.nzval[idx]

    # Substitute next values involving x[col]
    for i = A.colptr[col] : idx - 1
      x[A.rowval[i]] -= A.nzval[i] * x[col]
    end
  end
  return x
end

function backward_sub!(U::AbstractArray{<:UpperTriangular},x::PVector)
  map(U,own_values(x)) do U, x
    backward_sub!(U, x)
  end
end

# Smoother

"""
    struct GaussSeidelSmoother <: Gridap.Algebra.LinearSolver
    GaussSeidelSmoother(niter::Integer=1, ω::Real=1.0, type::Symbol=:symmetric)
"""
struct GaussSeidelSmoother <: Gridap.Algebra.LinearSolver
  niter :: Int
  ω     :: Float64
  type  :: Symbol

  function GaussSeidelSmoother(
    niter::Integer=1, ω::Real=1.0, type::Symbol=:symmetric
  )
    @check type in (:symmetric, :forward, :backward) "The type must be :symmetric, :forward or :backward."
    @check niter > 0 "The number of iterations must be a positive integer."
    @check ω > 0.0 "The relaxation parameter must be a positive real number."
    return new(niter,ω,type)
  end
end

# For backward compatibility
SymGaussSeidelSmoother(niter::Integer=1, ω::Real=1.0) = GaussSeidelSmoother(niter,ω,:symmetric)

struct GaussSeidelSymbolicSetup <: Gridap.Algebra.SymbolicSetup
  solver :: GaussSeidelSmoother
end

function Gridap.Algebra.symbolic_setup(s::GaussSeidelSmoother,A::AbstractMatrix)
  GaussSeidelSymbolicSetup(s)
end

# Numerical setup

struct GaussSeidelNumericalSetup{A,B,C,D} <: Gridap.Algebra.NumericalSetup
  solver :: GaussSeidelSmoother
  mat    :: A
  L      :: B
  U      :: C
  caches :: D
end

function _gs_get_caches(A::AbstractMatrix)
  dx  = allocate_in_domain(A)
  Adx = allocate_in_range(A)
  return dx, Adx
end

function _gs_decompose_matrix(A::AbstractMatrix)
  idx_range = 1:minimum(size(A))
  D  = DiagonalIndices(A,idx_range)
  L  = LowerTriangular(A, D)
  U  = UpperTriangular(A, D)
  return L, U
end

function _gs_decompose_matrix(A::PSparseMatrix{T}) where T
  values  = partition(A)
  indices = isa(PartitionedArrays.getany(values),SparseMatrixCSC) ? partition(axes(A,2)) : partition(axes(A,1))
  L,U = map(values,indices) do A, indices
    D = DiagonalIndices(A,own_to_local(indices))
    L = LowerTriangular(A,D)
    U = UpperTriangular(A,D)
    return L, U
  end |> tuple_of_arrays
  return L, U
end

function Gridap.Algebra.numerical_setup(ss::GaussSeidelSymbolicSetup,A::AbstractMatrix)
  L, U   = _gs_decompose_matrix(A)
  caches = _gs_get_caches(A)
  return GaussSeidelNumericalSetup(ss.solver,A,L,U,caches)
end

# Solve

function Gridap.Algebra.solve!(x::AbstractVector, ns::GaussSeidelNumericalSetup, r::AbstractVector)
  A, L, U, caches = ns.mat, ns.L, ns.U, ns.caches
  niter, ω = ns.solver.niter, ns.solver.ω
  dx, Adx = caches

  forward = ns.solver.type ∈ (:forward, :symmetric)
  backward = ns.solver.type ∈ (:backward, :symmetric)

  iter = 1
  while iter <= niter
    
    if forward # Forward pass
      copy!(dx,r)
      forward_sub!(L, dx)
      dx .= ω .* dx
      x  .= x .+ dx
      mul!(Adx, A, dx)
      r  .= r .- Adx
    end

    if backward # Backward pass
      copy!(dx,r)
      backward_sub!(U, dx)
      dx .= ω .* dx
      x  .= x .+ dx
      mul!(Adx, A, dx)
      r  .= r .- Adx
    end

    iter += 1
  end

  return x
end

function LinearAlgebra.ldiv!(x::AbstractVector, ns::GaussSeidelNumericalSetup, b::AbstractVector)
  fill!(x,0.0)
  aux = copy(b)
  solve!(x,ns,aux)
end
