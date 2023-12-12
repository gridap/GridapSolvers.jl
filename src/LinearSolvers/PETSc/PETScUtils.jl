
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

function interpolation_operator(op,U_in,V_out;
                                strat=SubAssembledRows(),
                                Tm=SparseMatrixCSR{0,PetscScalar,PetscInt},
                                Tv=Vector{PetscScalar})
  out_dofs = get_fe_dof_basis(V_out)
  in_basis  = get_fe_basis(U_in)
  
  cell_interp_mats = out_dofs(op(in_basis))
  local_contr = map(local_views(out_dofs),cell_interp_mats) do dofs, arr
    contr = DomainContribution()
    add_contribution!(contr,get_triangulation(dofs),arr)
    return contr
  end
  contr = GridapDistributed.DistributedDomainContribution(local_contr)
  
  matdata = collect_cell_matrix(U_in,V_out,contr)
  assem = SparseMatrixAssembler(Tm,Tv,U_in,V_out,strat)
  
  I = allocate_matrix(assem,matdata)
  takelast_matrix!(I,assem,matdata)
  return I
end

function takelast_matrix(a::SparseMatrixAssembler,matdata)
  m1 = Gridap.Algebra.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  symbolic_loop_matrix!(m1,a,matdata)
  m2 = Gridap.Algebra.nz_allocation(m1)
  takelast_loop_matrix!(m2,a,matdata)
  m3 = Gridap.Algebra.create_from_nz(m2)
  return m3
end

function takelast_matrix!(mat,a::SparseMatrixAssembler,matdata)
  LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
  takelast_matrix_add!(mat,a,matdata)
end

function takelast_matrix_add!(mat,a::SparseMatrixAssembler,matdata)
  takelast_loop_matrix!(mat,a,matdata)
  Gridap.Algebra.create_from_nz(mat)
end

function takelast_loop_matrix!(A,a::GridapDistributed.DistributedSparseMatrixAssembler,matdata)
  rows = get_rows(a)
  cols = get_cols(a)
  map(takelast_loop_matrix!,local_views(A,rows,cols),local_views(a),matdata)
end

function takelast_loop_matrix!(A,a::SparseMatrixAssembler,matdata)
  strategy = Gridap.FESpaces.get_assembly_strategy(a)
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = Gridap.FESpaces.map_cell_rows(strategy,_cellidsrows)
    cellidscols = Gridap.FESpaces.map_cell_cols(strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat) == length(cellidsrows)
    if length(cellmat) > 0
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      vals_cache = array_cache(cellmat)
      mat1 = getindex!(vals_cache,cellmat,1)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = Gridap.Arrays.AddEntriesMap((a,b) -> b)
      add_cache = return_cache(add!,A,mat1,rows1,cols1)
      caches = add_cache, vals_cache, rows_cache, cols_cache
      _takelast_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
    end
  end
  A
end

@noinline function _takelast_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols)
  add_cache, vals_cache, rows_cache, cols_cache = caches
  add! = Gridap.Arrays.AddEntriesMap((a,b) -> b)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end
