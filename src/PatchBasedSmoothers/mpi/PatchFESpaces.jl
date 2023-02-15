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

  function f(model,patch_decomposition,Vh,partition)
    patches_mask = fill(false,length(partition.lid_to_gid))
    patches_mask[partition.hid_to_lid] .= true # Mask ghost patch roots
    PatchFESpace(model,
                 reffe,
                 conformity,
                 patch_decomposition,
                 Vh;
                 patches_mask=patches_mask)
  end

  spaces = map_parts(f,
                   model.models,
                   patch_decomposition.patch_decompositions,
                   Vh.spaces,
                   root_gids.partition)
  
  parts  = get_part_ids(model.models)
  nodofs = map_parts(spaces) do space
    num_free_dofs(space)
  end
  ngdofs = sum(nodofs)

  first_gdof, _ = xscan(+,reduce,nodofs,init=1)
  # This PRange has no ghost dofs
  gids = PRange(parts,ngdofs,nodofs,first_gdof)
  return GridapDistributed.DistributedSingleFieldFESpace(spaces,gids,get_vector_type(Vh))
end

# x \in  PatchFESpace
# y \in  SingleFESpace
function prolongate!(x::PVector,
                     Ph::GridapDistributed.DistributedSingleFieldFESpace,
                     y::PVector)
   map_parts(x.values,Ph.spaces,y.values) do x,Ph,y
     prolongate!(x,Ph,y)
   end
end

function inject!(x::PVector,
                 Ph::GridapDistributed.DistributedSingleFieldFESpace,
                 y::PVector,
                 w::PVector,
                 w_sums::PVector)

  map_parts(x.values,Ph.spaces,y.values,w.values,w_sums.values) do x,Ph,y,w,w_sums
    inject!(x,Ph,y,w,w_sums)
  end

  # Exchange local contributions 
  assemble!(x)
  exchange!(x) # TO CONSIDER: Is this necessary? Do we need ghosts for later?
  return x
end

function compute_weight_operators(Ph::GridapDistributed.DistributedSingleFieldFESpace,Vh)
  # Local weights and partial sums
  w = PVector(0.0,Ph.gids)
  w_sums = PVector(0.0,Vh.gids)
  map_parts(w.values,w_sums.values,Ph.spaces) do w, w_sums, Ph
    _w, _w_sums = compute_weight_operators(Ph)
    w .= _w
    w_sums .= _w_sums
  end
  
  # partial sums -> global sums
  assemble!(w_sums) # ghost -> owners
  exchange!(w_sums) # repopulate ghosts with owner info

  return w, w_sums
end
