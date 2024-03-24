
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

function Base.map(f::Function,args::Vararg{HierarchicalArray,N}) where N
  ranks = get_level_parts(first(args))
  @assert all(a -> get_level_parts(a) === ranks, args)

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
