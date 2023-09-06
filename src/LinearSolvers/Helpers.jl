
# Row/Col vector allocations for serial
function allocate_row_vector(A::AbstractMatrix{T}) where T
  return zeros(T,size(A,1))
end

function allocate_col_vector(A::AbstractMatrix{T}) where T
  return zeros(T,size(A,2))
end

# Row/Col vector allocations for parallel
function allocate_row_vector(A::PSparseMatrix)
  T = eltype(A)
  return pfill(zero(T),partition(axes(A,1)))
end

function allocate_col_vector(A::PSparseMatrix)
  T = eltype(A)
  return pfill(zero(T),partition(axes(A,2)))
end

# Row/Col vector allocations for blocks
function allocate_row_vector(A::AbstractBlockMatrix)
  return mortar(map(Aii->allocate_row_vector(Aii),blocks(A)[:,1]))
end

function allocate_col_vector(A::AbstractBlockMatrix)
  return mortar(map(Aii->allocate_col_vector(Aii),blocks(A)[1,:]))
end
