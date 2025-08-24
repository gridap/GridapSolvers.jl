
"""
    CoarsePatchTopology(model::AdaptedDiscreteModel)

Given an `AdaptedDiscreteModel`, returns a `PatchTopology` for the fine model 
such that each patch corresponds to a coarse cell in the parent model.
"""
function CoarsePatchTopology(
  model::Gridap.Adaptivity.AdaptedDiscreteModel,
  coarse_ids = 1:num_cells(Gridap.Adaptivity.get_parent(model))
)
  ftopo = get_grid_topology(model)
  glue = Gridap.Adaptivity.get_adaptivity_glue(model)
  patch_cells = glue.o2n_faces_map[coarse_ids]
  return PatchTopology(ftopo,patch_cells)
end

function CoarsePatchTopology(model::GridapDistributed.DistributedDiscreteModel)
  parent = Gridap.Adaptivity.get_parent(model)
  cgids = get_cell_gids(parent)
  ptopos = map(local_views(model),partition(cgids)) do model, cgids
    CoarsePatchTopology(model,own_to_local(cgids))
  end
  GridapDistributed.DistributedPatchTopology(ptopos)
end
