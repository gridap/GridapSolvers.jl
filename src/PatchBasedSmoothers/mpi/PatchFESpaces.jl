# Rationale behind distributed PatchFESpace:
# 1. Patches have an owner. Only owners compute subspace correction.
#    If am not owner of a patch, all dofs in my patch become -1. [DONE]
# 2. Subspace correction on an owned patch may affect DoFs  which
#    are non-owned. These corrections should be sent to the owner
#    process. I.e., NO -> O (reversed) communication. [PENDING]
# 3. Each processor needs to know how many patches "touch" its owned DoFs.
#    This requires NO->O communication as well. [PENDING]

function PatchFESpace(model::GridapDistributed.DistributedDiscreteModel,
                      reffe::Tuple{<:Gridap.FESpaces.ReferenceFEName,Any,Any},
                      conformity::Gridap.FESpaces.Conformity,
                      patch_decomposition::DistributedPatchDecomposition,
                      Vh::GridapDistributed.DistributedSingleFieldFESpace)
  root_gids = get_face_gids(model,get_patch_root_dim(patch_decomposition))

  spaces = map(local_views(model),
                     local_views(patch_decomposition),
                     local_views(Vh),
                     root_gids.partition) do model, patch_decomposition, Vh, partition
    patches_mask = fill(false,length(partition.lid_to_gid))
    patches_mask[partition.hid_to_lid] .= true # Mask ghost patch roots
    PatchFESpace(model,reffe,conformity,patch_decomposition,Vh;patches_mask=patches_mask)
  end
  
  parts  = get_parts(model)
  local_ndofs  = map(num_free_dofs,spaces)
  global_ndofs = sum(local_ndofs)
  first_gdof, _ = xscan(+,reduce,local_ndofs,init=1)
  # This PRange has no ghost dofs
  gids = PRange(parts,global_ndofs,local_ndofs,first_gdof)
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
   map(x.values,Ph.spaces,y.values) do x,Ph,y
     prolongate!(x,Ph,y)
   end
   consistent!(x) |> fetch
end

# x \in  SingleFESpace
# y \in  PatchFESpace
function inject!(x::PVector,
                 Ph::GridapDistributed.DistributedSingleFieldFESpace,
                 y::PVector,
                 w::PVector,
                 w_sums::PVector)

  #consistent!(y)
  map(x.values,Ph.spaces,y.values,w.values,w_sums.values) do x,Ph,y,w,w_sums
    inject!(x,Ph,y,w,w_sums)
  end

  # Exchange local contributions 
  assemble!(x)
  consistent!(x) |> fetch # TO CONSIDER: Is this necessary? Do we need ghosts for later?
  return x
end

function compute_weight_operators(Ph::GridapDistributed.DistributedSingleFieldFESpace,Vh)
  # Local weights and partial sums
  w = PVector(0.0,Ph.gids)
  w_sums = PVector(0.0,Vh.gids)
  map(w.values,w_sums.values,Ph.spaces) do w, w_sums, Ph
    compute_weight_operators!(Ph,Ph.Vh,w,w_sums)
  end
  
  # partial sums -> global sums
  assemble!(w_sums) # ghost -> owners
  consistent!(w_sums) |> fetch # repopulate ghosts with owner info

  return w, w_sums
end
