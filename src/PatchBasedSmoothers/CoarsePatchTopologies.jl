
"""
    CoarsePatchTopology(model::AdaptedDiscreteModel)

Given an `AdaptedDiscreteModel`, returns a `PatchTopology` for the fine model 
such that each patch corresponds to a coarse cell in the parent model.
"""
function CoarsePatchTopology(
  model::Gridap.Adaptivity.AdaptedDiscreteModel,
  coarse_ids = 1:num_cells(Gridap.Adaptivity.get_parent(model));
  Dr = 0
)
  Dc = num_cell_dims(model)
  ftopo = get_grid_topology(model)
  ctopo = get_grid_topology(Gridap.Adaptivity.get_parent(model))
  glue = Gridap.Adaptivity.get_adaptivity_glue(model)

  # Each patch corresponds to a coarse cell. The fine patch cells are the 
  # children of the coarse cell in the adaptivity glue.
  patch_cells = glue.o2n_faces_map[coarse_ids]
  
  # The patch root is the unique r-fface that is interior to the coarse cell, 
  # i.e the one that has it's cface dimension equal to Dc.
  fcell_to_ffaces = Geometry.get_faces(ftopo,Dc,Dr)
  fnode_to_cface_dim = Gridap.Adaptivity.get_d_to_fface_to_cface(glue, ctopo, ftopo)[2][1]
  is_interior(n) = isequal(fnode_to_cface_dim[n],Dc)
  patch_roots = map(patch_cells) do cells
    faces = filter(is_interior,intersect((fcell_to_ffaces[c] for c in cells)...))
    return only(faces)
  end
  metadata = Geometry.StarPatchMetadata(Dr,patch_roots)

  return PatchTopology(ftopo,patch_cells,metadata)
end

function CoarsePatchTopology(model::GridapDistributed.DistributedDiscreteModel)
  parent = Gridap.Adaptivity.get_parent(model)
  cgids = get_cell_gids(parent)
  ptopos = map(local_views(model),partition(cgids)) do model, cgids
    CoarsePatchTopology(model,own_to_local(cgids))
  end
  GridapDistributed.DistributedPatchTopology(ptopos)
end
