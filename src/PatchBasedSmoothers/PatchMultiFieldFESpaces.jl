
## PatchFESpace from MultiFieldFESpace

@doc """
    function PatchFESpace(
      space::Gridap.MultiField.MultiFieldFESpace,
      patch_decomposition::PatchDecomposition,
      cell_conformity::Vector{<:CellConformity};
      kwargs...
    )

`PatchFESpace` constructor for `MultiFieldFESpace`. 
Returns a `MultiFieldFESpace` of `PatchFESpace`s .
"""
function PatchFESpace(
  space::Gridap.MultiField.MultiFieldFESpace,
  patch_decomposition::PatchDecomposition,
  cell_conformity::Vector{<:CellConformity};
  kwargs...
)
  patch_spaces = map((s,c) -> PatchFESpace(s,patch_decomposition,c;kwargs...),space,cell_conformity)
  return MultiFieldFESpace(patch_spaces)
end

function PatchFESpace(
  space::GridapDistributed.DistributedMultiFieldFESpace,
  patch_decomposition::DistributedPatchDecomposition,
  cell_conformity::Vector;
  patches_mask = default_patches_mask(patch_decomposition)
)

  field_spaces = map((s,c) -> PatchFESpace(s,patch_decomposition,c;patches_mask),space,cell_conformity)
  part_spaces = map(MultiFieldFESpace,GridapDistributed.to_parray_of_arrays(map(local_views,field_spaces)))
  
  # This PRange has no ghost dofs
  local_ndofs  = map(num_free_dofs,part_spaces)
  global_ndofs = sum(local_ndofs)
  patch_partition = variable_partition(local_ndofs,global_ndofs,false)
  gids = PRange(patch_partition)

  vector_type = get_vector_type(space)
  return GridapDistributed.DistributedMultiFieldFESpace(field_spaces,part_spaces,gids,vector_type)
end

# Inject/Prolongate for MultiField (only for ConsecutiveMultiFieldStyle)

# x \in  PatchFESpace
# y \in  SingleFESpace
function prolongate!(x,Ph::MultiFieldFESpace,y)
  Ph_spaces = Ph.spaces
  Vh_spaces = map(Phi -> Phi.Vh, Ph_spaces)
  Ph_offsets = Gridap.MultiField._compute_field_offsets(Ph_spaces)
  Vh_offsets = Gridap.MultiField._compute_field_offsets(Vh_spaces)
  Ph_ndofs = map(num_free_dofs,Ph_spaces)
  Vh_ndofs = map(num_free_dofs,Vh_spaces)
  for (i,Ph_i) in enumerate(Ph_spaces)
    x_i = view(x, Ph_offsets[i]+1:Ph_offsets[i]+Ph_ndofs[i])
    y_i = view(y, Vh_offsets[i]+1:Vh_offsets[i]+Vh_ndofs[i])
    prolongate!(x_i,Ph_i,y_i)
  end
end

# x \in  SingleFESpace
# y \in  PatchFESpace
function inject!(x,Ph::MultiFieldFESpace,y)
  Ph_spaces = Ph.spaces
  Vh_spaces = map(Phi -> Phi.Vh, Ph_spaces)
  Ph_offsets = Gridap.MultiField._compute_field_offsets(Ph_spaces)
  Vh_offsets = Gridap.MultiField._compute_field_offsets(Vh_spaces)
  Ph_ndofs = map(num_free_dofs,Ph_spaces)
  Vh_ndofs = map(num_free_dofs,Vh_spaces)
  for (i,Ph_i) in enumerate(Ph_spaces)
    y_i = view(y, Ph_offsets[i]+1:Ph_offsets[i]+Ph_ndofs[i])
    x_i = view(x, Vh_offsets[i]+1:Vh_offsets[i]+Vh_ndofs[i])
    inject!(x_i,Ph_i,y_i)
  end
end

function prolongate!(
  x::PVector,
  Ph::GridapDistributed.DistributedMultiFieldFESpace,
  y::PVector;
  is_consistent::Bool=false
)
  if !is_consistent
    consistent!(y) |> fetch
  end
  map(prolongate!,partition(x),local_views(Ph),partition(y))
end

function inject!(
  x::PVector,
  Ph::GridapDistributed.DistributedMultiFieldFESpace,
  y::PVector;
  make_consistent::Bool=true
)
  map(partition(x),local_views(Ph),partition(y)) do x,Ph,y
    inject!(x,Ph,y)
  end

  # Exchange local contributions 
  assemble!(x) |> fetch
  if make_consistent
    consistent!(x) |> fetch
  end
  return x
end
