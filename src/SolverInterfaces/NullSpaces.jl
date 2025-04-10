struct NullSpace{T,VT}
  V :: VT
  function NullSpace(
    V::AbstractVector{<:AbstractVector{T}}
  ) where T
    n = length(first(V))
    @assert all(length(v) == n for v in V)
    VT = typeof(V)
    new{T,VT}(V)
  end
end

Base.size(N::NullSpace) = (length(N.V),length(first(N.V)))
Base.size(N::NullSpace, i::Int) = size(N)[i]
Base.merge(a::NullSpace{T}, b::NullSpace{T}) where T = NullSpace(vcat(a.V, b.V))

function matrix_representation(N::NullSpace)
  return stack(N.V)
end

NullSpace(v::AbstractVector{<:Number}) = NullSpace([v])

function NullSpace(A::Matrix)
  V = eachcol(nullspace(A))
  return NullSpace(V)
end

function NullSpace(f::Function,X::FESpace,Λ::FESpace)
  A = assemble_matrix(f, X, Λ)
  return NullSpace(A)
end

function is_orthonormal(N::NullSpace; tol = 1.e-12)
  for w in N.V
    !(abs(norm(w) - 1.0) < tol) && return false
  end
  return is_orthogonal(N, tol = tol)
end

function is_orthogonal(N::NullSpace; tol = 1.e-12)
  for (k,w) in enumerate(N.V)
    for v in N.V[k+1:end]
      !(abs(dot(w, v)) < tol) && return false
    end
  end
  return true
end

function is_orthogonal(N::NullSpace, v::AbstractVector; tol = 1.e-12)
  @assert length(v) == size(N,2)
  for w in N.V
    !(abs(dot(w, v)) < tol) && return false
  end
  return true
end

function is_orthogonal(N::NullSpace, A::AbstractMatrix; tol = 1.e-12)
  @assert length(v) == size(N,2)
  v = allocate_in_range(A)
  for w in N.V
    mul!(v, A, w)
    !(abs(norm(v)) < tol) && return false
  end
  return true
end

function make_orthonormal!(N::NullSpace; method = :gram_schmidt)
  if method == :gram_schmidt
    gram_schmidt!(N.V)
  elseif method == :modified_gram_schmidt
    modified_gram_schmidt!(N.V)
  else
    error("Unknown method: $method")
  end
  return N
end

function gram_schmidt!(V::AbstractVector{<:AbstractVector{T}}) where T
  n = length(V)
  for j in 1:n
    for i in 1:j-1
      α = dot(V[j], V[i])
      V[j] .-= α .* V[i]
    end
    V[j] ./= norm(V[j])
  end
  return V
end

function modified_gram_schmidt!(V::AbstractVector{<:AbstractVector{T}}) where T
  n = length(V)
  for j in 1:n
    V[j] ./= norm(V[j])
    for i in j+1:n
      α = dot(V[j], V[i])
      V[i] .-= α .* V[j]
    end
  end
  return V
end

function project(N::NullSpace,v)
  p = similar(v)
  project!(p,N,v)
end

function project!(p,N::NullSpace{T},v) where T
  @assert length(v) == size(N,2)
  α = zeros(T, size(N,1))
  fill!(p,0.0)
  for (k,w) in enumerate(N.V)
    α[k] = dot(v, w)
    p .+= α[k] .* w
  end
  return p, α
end

function make_orthogonal!(N::NullSpace{T},v) where T
  @assert length(v) == size(N,2)
  α = zeros(T, size(N,1))
  for (k,w) in enumerate(N.V)
    α[k] = dot(v, w)
    v .-= α[k] .* w
  end
  return v, α
end

function reconstruct(N::NullSpace,v,α)
  w = copy(v)
  reconstruct!(N,w,α)
  return w
end

function reconstruct!(N::NullSpace,v,α)
  for (k,w) in enumerate(N.V)
    v .+= α[k] .* w
  end
  return v
end
