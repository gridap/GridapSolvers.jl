
# This could be a DistributedSingleFieldFESpace if it accepted all kinds of FESpaces
struct PatchDistributedMultiFieldFESpace{A,B}
  spaces :: A
  gids   :: B
end

GridapDistributed.local_views(a::PatchDistributedMultiFieldFESpace) = a.spaces

## PatchFESpace from MultiFieldFESpace

function PatchFESpace(space::Gridap.MultiField.MultiFieldFESpace,
                      patch_decomposition::PatchDecomposition,
                      cell_conformity::Vector{<:CellConformity};
                      kwargs...)
  patch_spaces = map((s,c) -> PatchFESpace(s,patch_decomposition,c;kwargs...),space,cell_conformity)
  return MultiFieldFESpace(patch_spaces)
end

function PatchFESpace(space::GridapDistributed.DistributedMultiFieldFESpace,
                      patch_decomposition::DistributedPatchDecomposition,
                      cell_conformity::Vector{<:AbstractArray{<:CellConformity}})
  model = patch_decomposition.model                    
  root_gids = get_face_gids(model,get_patch_root_dim(patch_decomposition))

  cell_conformity = GridapDistributed.to_parray_of_arrays(cell_conformity)
  spaces = map(local_views(space),
               local_views(patch_decomposition),
               cell_conformity,
               partition(root_gids)) do space, patch_decomposition, cell_conformity, partition
    patches_mask = fill(false,local_length(partition))
    patches_mask[ghost_to_local(partition)] .= true # Mask ghost patch roots
    PatchFESpace(space,patch_decomposition,cell_conformity;patches_mask=patches_mask)
  end
  
  # This PRange has no ghost dofs
  local_ndofs  = map(num_free_dofs,spaces)
  global_ndofs = sum(local_ndofs)
  patch_partition = variable_partition(local_ndofs,global_ndofs,false)
  gids = PRange(patch_partition)
  return PatchDistributedMultiFieldFESpace(spaces,gids)
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
    x_i = SubVector(x, Ph_offsets[i]+1, Ph_offsets[i] + Ph_ndofs[i])
    y_i = SubVector(y, Vh_offsets[i]+1, Vh_offsets[i] + Vh_ndofs[i])
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
    y_i = SubVector(y, Ph_offsets[i]+1, Ph_offsets[i] + Ph_ndofs[i])
    x_i = SubVector(x, Vh_offsets[i]+1, Vh_offsets[i] + Vh_ndofs[i])
    inject!(x_i,Ph_i,y_i)
  end
end

# Copied from PatchFESpaces, could be made redundant if DistributedSingleFieldFESpace was abstract

function prolongate!(x::PVector,
                     Ph::PatchDistributedMultiFieldFESpace,
                     y::PVector;
                     is_consistent::Bool=false)
  if !is_consistent
    consistent!(y) |> fetch
  end
  map(prolongate!,partition(x),local_views(Ph),partition(y))
end

function inject!(x::PVector,
                 Ph::PatchDistributedMultiFieldFESpace,
                 y::PVector;
                 make_consistent::Bool=true)

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
