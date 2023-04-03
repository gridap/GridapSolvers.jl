
# Row/Col vectors 

function allocate_row_vector(A::AbstractMatrix{T}) where T
  return zeros(T,size(A,1))
end

function allocate_col_vector(A::AbstractMatrix{T}) where T
  return zeros(T,size(A,2))
end


function allocate_row_vector(A::PSparseMatrix)
  T = eltype(A)
  return PVector(zero(T),A.rows)
end

function allocate_col_vector(A::PSparseMatrix)
  T = eltype(A)
  return PVector(zero(T),A.cols)
end


