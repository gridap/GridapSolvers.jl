
function PatchFESpace(space::GridapDistributed.DistributedSingleFieldFESpace,
                      patch_decomposition::DistributedPatchDecomposition,
                      reffe::Union{ReferenceFE,Tuple{<:Gridap.ReferenceFEs.ReferenceFEName,Any,Any}};
                      conformity=nothing)
  cell_conformity = MultilevelTools._cell_conformity(patch_decomposition.model,reffe;conformity=conformity)
  return PatchFESpace(space,patch_decomposition,cell_conformity)
end

function PatchFESpace(
  space::GridapDistributed.DistributedSingleFieldFESpace,
  patch_decomposition::DistributedPatchDecomposition,
  cell_conformity::AbstractArray{<:CellConformity};
  patches_mask = default_patches_mask(patch_decomposition)
)
  spaces = map(local_views(space),
               local_views(patch_decomposition),
               cell_conformity,
               patches_mask) do space, patch_decomposition, cell_conformity, patches_mask
    PatchFESpace(space,patch_decomposition,cell_conformity;patches_mask)
  end
  
  # This PRange has no ghost dofs
  local_ndofs  = map(num_free_dofs,spaces)
  global_ndofs = sum(local_ndofs)
  patch_partition = variable_partition(local_ndofs,global_ndofs,false)
  gids = PRange(patch_partition)
  return GridapDistributed.DistributedSingleFieldFESpace(spaces,gids,get_vector_type(space))
end

function default_patches_mask(patch_decomposition::DistributedPatchDecomposition)
  model = patch_decomposition.model
  root_gids = get_face_gids(model,get_patch_root_dim(patch_decomposition))
  patches_mask = map(partition(root_gids)) do partition
    patches_mask = fill(false,local_length(partition))
    patches_mask[ghost_to_local(partition)] .= true # Mask ghost patch roots
    return patches_mask
  end
  return patches_mask
end

function PatchFESpace(
  sh::FESpaceHierarchy,
  patch_decompositions::AbstractArray{<:DistributedPatchDecomposition}
)
  map(view(sh,1:num_levels(sh)-1),patch_decompositions) do shl,decomp
    space = get_fe_space(shl)
    cell_conformity = shl.cell_conformity
    patch_space = PatchFESpace(space,decomp,cell_conformity)
    MultilevelTools.FESpaceHierarchyLevel(lev,nothing,patch_space,cell_conformity)
  end
end

# x \in  PatchFESpace
# y \in  SingleFESpace
# x is always consistent at the end since Ph has no ghosts
function prolongate!(x::PVector,
                     Ph::GridapDistributed.DistributedSingleFieldFESpace,
                     y::PVector;
                     is_consistent::Bool=false)
  if is_consistent 
    map(prolongate!,partition(x),local_views(Ph),partition(y))
  else
    # Transfer ghosts while copying owned dofs
    rows = axes(y,1)
    t = consistent!(y)
    map(partition(x),local_views(Ph),partition(y),own_to_local(rows)) do x,Ph,y,ids
      prolongate!(x,Ph,y;dof_ids=ids)
    end
    # Wait for transfer to end and copy ghost dofs
    wait(t)
    map(partition(x),local_views(Ph),partition(y),ghost_to_local(rows)) do x,Ph,y,ids
      prolongate!(x,Ph,y;dof_ids=ids)
    end
  end
end

# x \in  SingleFESpace
# y \in  PatchFESpace
# y is always consistent at the start since Ph has no ghosts
function inject!(x::PVector,
                 Ph::GridapDistributed.DistributedSingleFieldFESpace,
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

function inject!(x::PVector,
                 Ph::GridapDistributed.DistributedSingleFieldFESpace,
                 y::PVector,
                 w::PVector,
                 w_sums::PVector;
                 make_consistent::Bool=true)

  map(partition(x),local_views(Ph),partition(y),partition(w),partition(w_sums)) do x,Ph,y,w,w_sums
    inject!(x,Ph,y,w,w_sums)
  end

  # Exchange local contributions 
  assemble!(x) |> fetch
  if make_consistent
    consistent!(x) |> fetch
  end
  return x
end

function compute_weight_operators(Ph::GridapDistributed.DistributedSingleFieldFESpace,Vh)
  # Local weights and partial sums
  w_values, w_sums_values = map(compute_weight_operators,local_views(Ph),local_views(Vh)) |> tuple_of_arrays
  w      = PVector(w_values,partition(Ph.gids))
  w_sums = PVector(w_sums_values,partition(Vh.gids))

  # partial sums -> global sums
  assemble!(w_sums) |> fetch   # ghost -> owners
  consistent!(w_sums) |> fetch # repopulate ghosts with owner info
  return w, w_sums
end
