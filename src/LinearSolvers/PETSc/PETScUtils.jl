
# DoF coordinates

"""
  Given a lagrangian FESpace, returns the physical coordinates of the DoFs, as required 
  by some PETSc solvers. See [PETSc documentation](https://petsc.org/release/manualpages/PC/PCSetCoordinates.html).
"""
function get_dof_coordinates(space::GridapDistributed.DistributedSingleFieldFESpace)
  coords  = map(local_views(space),partition(space.gids)) do space, dof_ids
    local_to_own_dofs = local_to_own(dof_ids)
    return get_dof_coordinates(space;perm=local_to_own_dofs)
  end

  ngdofs  = length(space.gids)
  indices = map(local_views(space.gids)) do dof_indices
    owner = part_id(dof_indices)
    own_indices   = OwnIndices(ngdofs,owner,own_to_global(dof_indices))
    ghost_indices = GhostIndices(ngdofs,Int64[],Int32[]) # We only consider owned dofs
    OwnAndGhostIndices(own_indices,ghost_indices)   
  end
  return PVector(coords,indices)
end

function get_dof_coordinates(space::FESpace;perm=Base.OneTo(num_free_dofs(space)))
  trian = get_triangulation(space)
  cell_dofs = get_fe_dof_basis(space)
  cell_ids  = get_cell_dof_ids(space)

  cell_ref_nodes = lazy_map(get_nodes,CellData.get_data(cell_dofs))
  cell_dof_to_node = lazy_map(get_dof_to_node,CellData.get_data(cell_dofs))
  cell_dof_to_comp = lazy_map(get_dof_to_comp,CellData.get_data(cell_dofs))

  cmaps = get_cell_map(trian)
  cell_phys_nodes = lazy_map(evaluate,cmaps,cell_ref_nodes)

  node_coords = Vector{Float64}(undef,maximum(perm))
  cache_nodes = array_cache(cell_phys_nodes)
  cache_ids = array_cache(cell_ids)
  cache_dof_to_node = array_cache(cell_dof_to_node)
  cache_dof_to_comp = array_cache(cell_dof_to_comp)
  for cell in 1:num_cells(trian)
    ids = getindex!(cache_ids,cell_ids,cell)
    nodes = getindex!(cache_nodes,cell_phys_nodes,cell)
    dof_to_comp = getindex!(cache_dof_to_comp,cell_dof_to_comp,cell)
    dof_to_node = getindex!(cache_dof_to_node,cell_dof_to_node,cell)
    for (dof,c,n) in zip(ids,dof_to_comp,dof_to_node)
      if (dof > 0) && (perm[dof] > 0)
        node_coords[perm[dof]] = nodes[n][c]
      end
    end
  end
  return node_coords
end

# Interpolation matrices

