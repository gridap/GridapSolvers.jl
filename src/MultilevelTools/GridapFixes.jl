
# Necessary when integrating the result from LocalProjectionMaps onto MultiFieldFEBasisComponents 
# in parallel. In general, this would also get triggered when doing a change_domain operation
# between two different Triangulation views.
# It has to do with blocks and how `extend` and `pos_neg_data` are getting dispatched. 
function Geometry._similar_empty(val::Fields.LinearCombinationFieldVector)
  #Fields.VoidBasis(val,true)
  values = zeros(eltype(val.values),size(val.values))
  Gridap.Fields.LinearCombinationFieldVector(values,val.fields)
end

# This below is another attempt to fix the issue. 
# I'm close, but doesn't work. 
"""
function Geometry._similar_empty(val::ArrayBlock{<:AbstractArray{<:Field},N}) where N
  T = typeof(Geometry._similar_empty(testitem(val.array)))
  array = Array{T,N}(undef,size(val.array))
  touched = copy(val.touched)
  a = ArrayBlock(array,touched)
  for i in eachindex(a)
    if a.touched[i]
      a.array[i] = Geometry._similar_empty(val.array[i])
    end
  end
  a
end

function Gridap.Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:ArrayBlock{<:AbstractArray{<:Field}}},i_to_iposneg::PosNegPartition
)
  nineg = length(i_to_iposneg.ineg_to_i)
  void = Geometry._similar_empty(testitem(ipos_to_val))
  ineg_to_val = Fill(void,nineg)
  _ipos_to_val = lazy_map(Broadcasting(Fields.VoidBasisMap(false)),ipos_to_val)
  _ipos_to_val, ineg_to_val
end

function Arrays.lazy_map(k::Broadcasting{<:Fields.VoidBasisMap},a::Arrays.LazyArray{<:Fill{<:Fields.BlockMap}})
  args = map(ai -> lazy_map(k.f, ai), a.args)
  lazy_map(a.maps.value,args)
end
"""

############################################################################################
# New API: get_cell_conformity

using Gridap.FESpaces: CellConformity, NodeToDofGlue
using Gridap.TensorValues: change_eltype

function get_cell_polytopes(trian::Triangulation)
  reffes = get_reffes(trian)
  polys  = map(get_polytope,reffes)
  ctypes = get_cell_type(trian)
  return expand_cell_data(polys,ctypes)
end

function get_cell_conformity(space::UnconstrainedFESpace{V,<:CellConformity}) where V
  return space.metadata
end

function get_cell_conformity(space::UnconstrainedFESpace{V,<:NodeToDofGlue{T}}) where {V,T}
  cell_polys = get_cell_polytopes(get_triangulation(space))
  polys, ctypes = compress_cell_data(cell_polys)
  reffes = map(p -> LagrangianRefFE(change_eltype(T,eltype(V)),p,1), polys)
  cell_reffe = expand_cell_data(reffes,ctypes)
  return CellConformity(cell_reffe,H1Conformity())
end

for ST in [:TrialFESpace,ZeroMeanFESpace,FESpaceWithConstantFixed]
  @eval begin 
    function get_cell_conformity(space::$ST)
      return get_cell_conformity(space.space)
    end
  end
end

function get_cell_conformity(space::MultiFieldFESpace)
  map(get_cell_conformity,space)
end

function get_cell_conformity(space::GridapDistributed.DistributedFESpace)
  map(get_cell_conformity,local_views(space))
end

function get_cell_conformity(space::GridapDistributed.DistributedMultiFieldFESpace)
  map(get_cell_conformity,space)
end
