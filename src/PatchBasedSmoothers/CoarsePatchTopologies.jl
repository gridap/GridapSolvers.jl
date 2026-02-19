
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

function CoarsePatchTopology(mhl::MultilevelTools.ModelHierarchyLevel)
  model = get_model_before_redist(mhl)
  ptopo = !isnothing(model) ? CoarsePatchTopology(model) : nothing
  if has_redistribution(mhl)
    ptopo = redistribute_patch_topology(ptopo,model,get_model(mhl),mhl.red_glue)
  end
  return ptopo
end

# The following assumes:
#  1 - each patch is fully locally contained in some processor in the new model
#  2 - If a patch is contained in more than one processor, it is fully contained in all of them
#  3 - no overlap between patches
# The above 1 and 2 conditions should be guaranteed for the typical star-patches we are using since 
# the maximum span of a star patch is 2.
function redistribute_patch_topology(ptopo,model,new_model,glue)
  parts = get_parts(glue)
  n_lcells = change_parts(
    !isnothing(ptopo) ? map(num_cells, local_views(model)) : nothing,
    parts; default = 0
  )
  n_lcells_new = map(num_cells, local_views(new_model))
  cell_indices, red_cell_indices = GridapDistributed.redistribute_indices(
    partition(get_cell_gids(model)), 
    map(n -> identity_table(Int32,Int32,n), n_lcells),
    map(n -> identity_table(Int32,Int32,n), n_lcells_new),
    new_model, glue; reverse=false
  )

  # Map patch lids to patch gids
  patch_to_cells = change_parts(
    !isnothing(ptopo) ? map(Geometry.get_patch_cells, local_views(ptopo)) : nothing,
    parts; default = empty_table(Int32,Int32,0)
  )
  n_own = map(length, patch_to_cells)
  first_patch = scan(+,n_own,type=:exclusive,init=one(Int))
  patch_gids = variable_partition(n_own, sum(n_own); start = first_patch)

  # Communicate 
  cell_to_lpatch = map((pcells,nc) -> Arrays.inverse_table(pcells, nc).data, patch_to_cells, n_lcells)
  cell_to_gpatch = map(getindex,patch_gids,cell_to_lpatch)
  t = redistribute(PVector(cell_to_gpatch,cell_indices), red_cell_indices)
  new_cell_to_gpatch = partition(fetch(t))

  # Build new PatchTopology
  new_cell_indices = partition(get_cell_gids(new_model))
  patch_to_new_cells = map(new_cell_to_gpatch,new_cell_indices) do new_cell_to_gpatch, cgids
    rank = part_id(cgids)
    lpatch_to_new_cells = [findall(isequal(p),new_cell_to_gpatch) for p in unique(new_cell_to_gpatch)]
    loc_to_own = local_to_own(cgids)
    loc_to_owner = local_to_owner(cgids)
    function is_own_patch(cells)
      A = all(!iszero,loc_to_own[cells])
      B = maximum(loc_to_owner[cells]) == rank
      return A || B
    end
    own_patches = findall(is_own_patch, lpatch_to_new_cells)
    return Table(lpatch_to_new_cells[own_patches])
  end
  new_ptopos = map(PatchTopology,local_views(get_grid_topology(new_model)),patch_to_new_cells)
  return GridapDistributed.DistributedPatchTopology(new_ptopos)
end
