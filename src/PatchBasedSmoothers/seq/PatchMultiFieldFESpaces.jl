
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


## MultiFieldFESpace from PatchFESpaces
#
#function Gridap.MultiField.MultiFieldFESpace(spaces::Vector{<:PatchFESpace})
#  return PatchMultiFieldFESpace(spaces)
#end
#
#function Gridap.MultiField.MultiFieldFESpace(spaces::Vector{<:GridapDistributed.DistributedSingleFieldFESpace{<:AbstractArray{T}}}) where T <: PatchFESpace
#  return PatchMultiFieldFESpace(spaces)
#end
#
## MultiField API
#
#function Gridap.FESpaces.get_cell_dof_ids(f::PatchMultiFieldFESpace,trian::Triangulation)
#  offsets = Gridap.MultiField._compute_field_offsets(f)
#  nfields = length(f.spaces)
#  active_block_data = Any[]
#  for i in 1:nfields
#    cell_dofs_i = get_cell_dof_ids(f.spaces[i],trian)
#    if i == 1
#      push!(active_block_data,cell_dofs_i)
#    else
#      offset = Int32(offsets[i])
#      o = Fill(offset,length(cell_dofs_i))
#      cell_dofs_i_b = lazy_map(Broadcasting(Gridap.MultiField._sum_if_first_positive),cell_dofs_i,o)
#      push!(active_block_data,cell_dofs_i_b)
#    end
#  end
#  return lazy_map(BlockMap(nfields,active_block_ids),active_block_data...)
#end
#
#function Gridap.FESpaces.get_fe_basis(f::PatchMultiFieldFESpace)
#  nfields = length(f.spaces)
#  all_febases = MultiFieldFEBasisComponent[]
#  for field_i in 1:nfields
#    dv_i = get_fe_basis(f.spaces[field_i])
#    @assert BasisStyle(dv_i) == TestBasis()
#    dv_i_b = MultiFieldFEBasisComponent(dv_i,field_i,nfields)
#    push!(all_febases,dv_i_b)
#  end
#  MultiFieldCellField(all_febases)
#end
#
#function Gridap.FESpaces.get_trial_fe_basis(f::PatchMultiFieldFESpace)
#  nfields = length(f.spaces)
#  all_febases = MultiFieldFEBasisComponent[]
#  for field_i in 1:nfields
#    du_i = get_trial_fe_basis(f.spaces[field_i])
#    @assert BasisStyle(du_i) == TrialBasis()
#    du_i_b = MultiFieldFEBasisComponent(du_i,field_i,nfields)
#    push!(all_febases,du_i_b)
#  end
#  MultiFieldCellField(all_febases)
#end
#