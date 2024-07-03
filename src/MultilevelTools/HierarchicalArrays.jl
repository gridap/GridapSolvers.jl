
"""
    HierarchicalArray{T,A,B} <: AbstractVector{T}

  Array of hierarchical (nested) distributed objects.
  Each level might live in a different subcommunicator. If a processor does not belong to 
  subcommunicator `ranks[i]`, then `array[i]` is `nothing`.
  
  However, it assumes: 
    - The subcommunicators are nested, so that `ranks[i]` contains `ranks[i+1]`.
    - The first subcommunicator does not have empty parts.
"""
struct HierarchicalArray{T,A,B} <: AbstractVector{T}
  array :: A
  ranks :: B
  function HierarchicalArray{T}(array::AbstractVector,ranks::AbstractVector) where T
    @assert length(array) == length(ranks)
    A = typeof(array)
    B = typeof(ranks)
    new{T,A,B}(array,ranks)
  end
end

function HierarchicalArray(array,ranks)
  T = typejoin(filter(t -> t != Nothing, map(typeof,array))...)
  HierarchicalArray{T}(array,ranks)
end

function HierarchicalArray{T}(::UndefInitializer,ranks::AbstractVector) where T
  array = Vector{Union{Nothing,T}}(undef,length(ranks))
  HierarchicalArray{T}(array,ranks)
end

Base.length(a::HierarchicalArray) = length(a.array)
Base.size(a::HierarchicalArray) = (length(a),)

function Base.getindex(a::HierarchicalArray,i::Integer)
  msg = "Processor does not belong to subcommunicator $i."
  @assert i_am_in(a.ranks[i]) msg
  a.array[i]
end

function Base.setindex!(a::HierarchicalArray,v,i::Integer)
  msg = "Processor does not belong to subcommunicator $i."
  @assert i_am_in(a.ranks[i]) msg
  a.array[i] = v
  return v
end

# Unsafe getindex: Returns the value without 
# checking if the processor belongs to the subcommunicator.
unsafe_getindex(a::HierarchicalArray,i::Integer) = getindex(a.array,i)
unsafe_getindex(a::AbstractArray,i::Integer) = getindex(a,i)

function Base.view(a::HierarchicalArray{T},I) where T
  return HierarchicalArray{T}(view(a.array,I),view(a.ranks,I))
end

Base.IndexStyle(::Type{HierarchicalArray}) = IndexLinear()

function PartitionedArrays.linear_indices(a::HierarchicalArray)
  ids = LinearIndices(a.array)
  return HierarchicalArray{eltype(ids)}(ids,a.ranks)
end

function Base.show(io::IO,k::MIME"text/plain",data::HierarchicalArray{T}) where T
  println(io,"HierarchicalArray{$T}")
end

num_levels(a::HierarchicalArray) = length(a.ranks)
get_level_parts(a::HierarchicalArray) = a.ranks
get_level_parts(a::HierarchicalArray,lev) = a.ranks[lev]

function matching_level_parts(a::HierarchicalArray,b::HierarchicalArray)
  @assert num_levels(a) == num_levels(b)
  return all(map(===, get_level_parts(a), get_level_parts(b)))
end

function matching_level_parts(arrays::Vararg{HierarchicalArray,N}) where N
  a1 = first(arrays)
  return all(a -> matching_level_parts(a1,a), arrays)
end

"""
    Base.map(f::Function,args::Vararg{HierarchicalArray,N}) where N

  Maps a function to a set of `HierarchicalArrays`. The function is applied only in the
  subcommunicators where the processor belongs to.
"""
function Base.map(f,args::Vararg{HierarchicalArray,N}) where N
  @assert matching_level_parts(args...)
  ranks  = get_level_parts(first(args))
  arrays = map(a -> a.array, args)
  array = map(ranks, arrays...) do ranks, arrays...
    if i_am_in(ranks)
      f(arrays...)
    else
      nothing
    end
  end
  return HierarchicalArray(array,ranks)
end

function Base.map(f,a::HierarchicalArray)
  array = map(a.ranks, a.array) do ranks, ai
    if i_am_in(ranks)
      f(ai)
    else
      nothing
    end
  end
  return HierarchicalArray(array,a.ranks)
end

function Base.map!(f,a::HierarchicalArray,args::Vararg{HierarchicalArray,N}) where N
  @assert matching_level_parts(a,args...)
  ranks  = get_level_parts(a)
  arrays = map(a -> a.array, args)
  map(ranks, a.array, arrays...) do ranks, ai, arrays...
    if i_am_in(ranks)
      ai = f(arrays...)
    else
      nothing
    end
  end
  return a
end

"""
    with_level(f::Function,a::HierarchicalArray,lev::Integer;default=nothing)
  
  Applies a function to the `lev`-th level of a `HierarchicalArray`. If the processor does not
  belong to the subcommunicator of the `lev`-th level, then `default` is returned.
"""
function with_level(f::Function,a::HierarchicalArray,lev::Integer;default=nothing)
  if i_am_in(a.ranks[lev])
    return f(a.array[lev])
  else
    return default
  end
end

function with_level(f::Function,a::AbstractArray,lev::Integer;default=nothing)
  f(a.array[lev])
end
