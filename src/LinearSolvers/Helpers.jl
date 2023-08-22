
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
function allocate_row_vector(A::BlockMatrix)
  return mortar(map(Aii->allocate_row_vector(Aii),blocks(A)[:,1]))
end

function allocate_col_vector(A::BlockMatrix)
  return mortar(map(Aii->allocate_col_vector(Aii),blocks(A)[1,:]))
end

# BlockArrays of PVectors/PSparseMatrices

const BlockPVector{T} = BlockVector{T,<:Vector{<:PVector{T}}}
const BlockPSparseMatrix{T,V} = BlockMatrix{T,<:Matrix{<:PSparseMatrix{V}}}

# BlockVector algebra 
function LinearAlgebra.mul!(y::BlockVector,A::BlockMatrix,x::BlockVector)
  o = one(eltype(A))
  for i in blockaxes(A,2)
    fill!(y[i],0.0)
    for j in blockaxes(A,2)
      mul!(y[i],A[i,j],x[j],o,o)
    end
  end
end

function LinearAlgebra.dot(x::BlockPVector,y::BlockPVector)
  return sum(map(dot,blocks(x),blocks(y)))
end

function Base.zero(v::BlockPVector)
  return mortar(map(zero,blocks(v)))
end

function Base.similar(v::BlockPVector)
  return mortar(map(similar,blocks(v)))
end

function LinearAlgebra.norm(v::BlockPVector)
  block_norms = map(norm,blocks(v))
  return sqrt(sum(block_norms.^2))
end

function Base.copyto!(y::BlockPVector,x::BlockPVector)
  @check blocklength(x) == blocklength(y)
  for i in blockaxes(x,1)
    copyto!(y[i],x[i])
  end
end

# BlockVector Broadcasting for PVectors

struct BlockPBroadcasted{A,B}
  blocks :: A
  axes   :: B
end

BlockArrays.blocks(b::BlockPBroadcasted) = b.blocks
BlockArrays.blockaxes(b::BlockPBroadcasted) = b.axes

function Base.broadcasted(f, args::Union{BlockPVector,BlockPBroadcasted}...)
  a1 = first(args)
  @boundscheck @assert all(ai -> blockaxes(ai) == blockaxes(a1),args)
  
  blocks_in = map(blocks,args)
  blocks_out = map((largs...)->Base.broadcasted(f,largs...),blocks_in...)
  
  return BlockPBroadcasted(blocks_out,blockaxes(a1))
end

function Base.broadcasted(f, a::Number, b::Union{BlockPVector,BlockPBroadcasted})
  blocks_out = map(b->Base.broadcasted(f,a,b),blocks(b))
  return BlockPBroadcasted(blocks_out,blockaxes(b))
end

function Base.broadcasted(f, a::Union{BlockPVector,BlockPBroadcasted}, b::Number)
  blocks_out = map(a->Base.broadcasted(f,a,b),blocks(a))
  return BlockPBroadcasted(blocks_out,blockaxes(a))
end

function Base.broadcasted(f,
                        a::Union{BlockPVector,BlockPBroadcasted},
                        b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,a,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  b::Union{BlockPVector,BlockPBroadcasted})
  Base.broadcasted(f,Base.materialize(a),b)
end

function Base.materialize(b::BlockPBroadcasted)
  blocks_out = map(Base.materialize,blocks(b))
  return mortar(blocks_out)
end

function Base.materialize!(a::BlockPVector,b::BlockPBroadcasted)
  map(Base.materialize!,blocks(a),blocks(b))
  return a
end
