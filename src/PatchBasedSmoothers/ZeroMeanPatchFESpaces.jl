
const ZeroMeanPatchFESpace{CA,S,B} = PatchFESpace{ZeroMeanFESpace{CA,S},B}

function PatchFESpace(
  space::FESpaces.ZeroMeanFESpace,
  patch_decomposition::PatchDecomposition,
  reffe::Union{ReferenceFE,Tuple{<:ReferenceFEs.ReferenceFEName,Any,Any}};
  conformity=nothing,
  patches_mask = Fill(false,num_patches(patch_decomposition))
)
  pspace = PatchFESpace(space.space.space,patch_decomposition,reffe;conformity,patches_mask)

  n_patches = num_patches(patch_decomposition)
  n_pdofs = pspace.num_dofs
  patch_cells = patch_decomposition.patch_cells
  pcell_to_pdofs = pspace.patch_cell_dofs_ids
  dof_to_pdof = pspace.dof_to_pdof

  dof_to_dvol = space.vol_i
  pdof_to_dvol = fill(0.0,n_pdofs)
  for (dof,pdofs) in enumerate(dof_to_pdof)
    pdof_to_dvol[pdofs] .= dof_to_dvol[dof]
  end

  patch_vol = fill(0.0,n_patches)
  pdof_to_new_pdof = fill(0,n_pdofs)
  n_pdofs_free = 0
  n_pdofs_fixed = 1 # -1 reserved for homogeneous dirichlet on filtered patches
  for patch in 1:n_patches
    if patches_mask[patch]
      continue
    end
    cell_s = patch_cells.ptrs[patch]
    cell_e = patch_cells.ptrs[patch+1]-1
    pdof_s = pcell_to_pdofs.ptrs[cell_s]
    pdof_e = pcell_to_pdofs.ptrs[cell_e+1]-1

    # Patch volume
    patch_vol[patch] = sum(pdof_to_dvol[pcell_to_pdofs.data[pdof_s:pdof_e]])
    
    # First pdof per patch is fixed
    pdof = pcell_to_pdofs.data[pdof_s]
    n_pdofs_fixed += 1
    pdof_to_new_pdof[pdof] = -n_pdofs_fixed
    pcell_to_pdofs.data[pdof_s] = -n_pdofs_fixed

    # The rest is free
    for k in pdof_s+1:pdof_e
      pdof = pcell_to_pdofs.data[k]
      n_pdofs_free += 1
      pdof_to_new_pdof[pdof] = n_pdofs_free
      pcell_to_pdofs.data[k] = n_pdofs_free
    end
  end

  dof_to_new_pdof = Table(collect(pdof_to_new_pdof[dof_to_pdof.data]), dof_to_pdof.ptrs)

  metadata = (patch_vol, pdof_to_dvol, n_pdofs, n_pdofs_free, n_pdofs_fixed, patches_mask)
  return PatchFESpace(space,patch_decomposition,n_pdofs_free,pcell_to_pdofs,dof_to_new_pdof,metadata)
end

# x \in  PatchFESpace
# y \in  SingleFESpace
function prolongate!(x,Ph::ZeroMeanPatchFESpace,y;dof_ids=LinearIndices(y))
  dof_to_pdof = Ph.dof_to_pdof
  fixed_dof = Ph.Vh.space.dof_to_fix
  
  ptrs = dof_to_pdof.ptrs
  data = dof_to_pdof.data
  z = VectorWithEntryInserted(y,fixed_dof,zero(eltype(x)))
  for dof in dof_ids
    for k in ptrs[dof]:ptrs[dof+1]-1
      pdof = data[k]
      if pdof > 0 # Is this correct? Should we correct the values in each patch? 
        x[pdof] = z[dof]
      end
    end
  end
end

# x \in  SingleFESpace
# y \in  PatchFESpace
function inject!(x,Ph::ZeroMeanPatchFESpace,y)
  dof_to_pdof = Ph.dof_to_pdof
  fixed_vals, _ = _compute_new_patch_fixedvals(y,Ph)
  fixed_dof = Ph.Vh.space.dof_to_fix
  
  ptrs = dof_to_pdof.ptrs
  data = dof_to_pdof.data
  z = VectorWithEntryInserted(x,fixed_dof,zero(eltype(x)))
  for dof in 1:length(dof_to_pdof)
    z[dof] = 0.0
    for k in ptrs[dof]:ptrs[dof+1]-1
      pdof = data[k]
      if pdof > 0
        z[dof] += y[pdof]
      elseif pdof != -1
        z[dof] += fixed_vals[-pdof-1]
      end
    end
  end

  x .-= z.value
  return x
end

function _compute_new_patch_fixedvals(x,Ph::ZeroMeanPatchFESpace)
  patch_vol, pdof_to_dvol, n_pdofs, n_pdofs_free, n_pdofs_fixed, patches_mask = Ph.metadata
  PD = Ph.patch_decomposition
  patch_cells = PD.patch_cells
  pcell_to_pdofs = Ph.patch_cell_dofs_ids

  n_patches = num_patches(PD)
  k = 0
  fixed_vals = fill(0.0,n_pdofs_fixed)
  fixed_pdofs = fill(0,n_pdofs_fixed)
  for patch in 1:n_patches
    if patches_mask[patch]
      continue
    end
    pcell_s = patch_cells.ptrs[patch]
    pcell_e = patch_cells.ptrs[patch+1]-1

    pdof_s = pcell_to_pdofs.ptrs[pcell_s]
    pdof_e = pcell_to_pdofs.ptrs[pcell_e+1]-1
    
    fixed_pdof = -pcell_to_pdofs.data[pdof_s]
    free_pdofs = pcell_to_pdofs.data[pdof_s+1]:pcell_to_pdofs.data[pdof_e]
    pdofs = (pcell_to_pdofs.data[pdof_s+1]:pcell_to_pdofs.data[pdof_e]+1) .+ k

    fv = view(x,free_pdofs)
    dv = [zero(eltype(fv))]
    vol_i = view(pdof_to_dvol,pdofs)
    c = FESpaces._compute_new_fixedval(fv,dv,vol_i,patch_vol[patch],1)
    
    k += 1
    x[free_pdofs] .+= c
    fixed_vals[k] = c
    fixed_pdofs[k] = fixed_pdof
  end

  return fixed_vals, fixed_pdofs
end
