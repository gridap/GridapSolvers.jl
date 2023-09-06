
function PatchFESpace(model::GridapDistributed.DistributedDiscreteModel,
                      reffe::Tuple{<:Gridap.FESpaces.ReferenceFEName,Any,Any},
                      conformity::Gridap.FESpaces.Conformity,
                      patch_decomposition::DistributedPatchDecomposition,
                      Vh::GridapDistributed.DistributedSingleFieldFESpace)
  root_gids = get_face_gids(model,get_patch_root_dim(patch_decomposition))

  spaces = map(local_views(model),
               local_views(patch_decomposition),
               local_views(Vh),
               partition(root_gids)) do model, patch_decomposition, Vh, partition
    patches_mask = fill(false,local_length(partition))
    patches_mask[ghost_to_local(partition)] .= true # Mask ghost patch roots
    PatchFESpace(model,reffe,conformity,patch_decomposition,Vh;patches_mask=patches_mask)
  end
  
  # This PRange has no ghost dofs
  local_ndofs  = map(num_free_dofs,spaces)
  global_ndofs = sum(local_ndofs)
  patch_partition = variable_partition(local_ndofs,global_ndofs,false)
  gids = PRange(patch_partition)
  return GridapDistributed.DistributedSingleFieldFESpace(spaces,gids,get_vector_type(Vh))
end

function PatchFESpace(mh::ModelHierarchy,
                      reffe::Tuple{<:Gridap.FESpaces.ReferenceFEName,Any,Any},
                      conformity::Gridap.FESpaces.Conformity,
                      patch_decompositions::AbstractArray{<:DistributedPatchDecomposition},
                      sh::FESpaceHierarchy)
  nlevs = num_levels(mh)
  levels = Vector{MultilevelTools.FESpaceHierarchyLevel}(undef,nlevs)
  for lev in 1:nlevs-1
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model  = get_model(mh,lev)
      space  = MultilevelTools.get_fe_space(sh,lev)
      decomp = patch_decompositions[lev]
      patch_space = PatchFESpace(model,reffe,conformity,decomp,space)
      levels[lev] = MultilevelTools.FESpaceHierarchyLevel(lev,nothing,patch_space)
    end
  end
  return FESpaceHierarchy(mh,levels)
end

# x \in  PatchFESpace
# y \in  SingleFESpace
function prolongate!(x::PVector,
                     Ph::GridapDistributed.DistributedSingleFieldFESpace,
                     y::PVector)
  map(partition(x),local_views(Ph),partition(y)) do x,Ph,y
    prolongate!(x,Ph,y)
  end
end

# x \in  SingleFESpace
# y \in  PatchFESpace
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
