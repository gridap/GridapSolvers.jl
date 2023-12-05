
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
  dx  = allocate_in_domain(A)
  Adx = allocate_in_range(A)
  return dx, Adx
end

function _gs_decompose_matrix(A::AbstractMatrix)
  idx_range = 1:minimum(size(A))
  D  = DiagonalIndices(A,idx_range)
  L  = LowerTriangular(A, D)
  U  = UpperTriangular(A, D)
  return L,U
end

function _gs_decompose_matrix(A::PSparseMatrix{T,<:AbstractArray{MatType}}) where {T, MatType}
  values  = partition(A)
  indices = isa(PartitionedArrays.getany(values),SparseMatrixCSC) ? partition(axes(A,2)) : partition(axes(A,1))
  L,U = map(values,indices) do A, indices
    D = DiagonalIndices(A,own_to_local(indices))
    L = LowerTriangular(A,D)
    U = UpperTriangular(A,D)
    return L,U
  end |> tuple_of_arrays
  return L,U
end

function Gridap.Algebra.numerical_setup(ss::SymGaussSeidelSymbolicSetup,A::AbstractMatrix)
  L, U   = _gs_decompose_matrix(A)
  caches = _gs_get_caches(A)
  return SymGaussSeidelNumericalSetup(ss.solver,A,L,U,caches)
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
