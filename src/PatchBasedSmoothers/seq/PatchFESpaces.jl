# INPUT
# [[1, 2]]
# [[1, 2], [2, 3]]
# [[2, 3], [3, 4]]
# [[3, 4], [4, 5]]
# [[4, 5]]

# OUTPUT
# [[1, 2]]
# [[3, 4], [4, 5]]
# [[6, 7], [7, 8]]
# [[9, 10], [10, 11]]
# [[12, 13]]

# Negative numbers correspond to Dirichlet DoFs
# in the GLOBAL space. In these examples, we
# are neglecting Dirichlet DoFs in the boundary
# of the patches (assuming they are needed)

# INPUT
# [[-1, 1]]
# [[-1, 1], [1, 2]]
# [[1, 2], [2, 3]]
# [[2, 3], [3, -2]]
# [[3, -2]]

# OUTPUT
# [[-1, 1]]
# [[-1, 2], [2, 3]]
# [[4, 5], [5, 6]]
# [[6, 7], [7, -2]]
# [[8, -2]]

struct PatchFESpace  <: Gridap.FESpaces.SingleFieldFESpace
  Vh                  :: Gridap.FESpaces.SingleFieldFESpace
  patch_decomposition :: PatchDecomposition
  num_dofs            :: Int
  patch_cell_dofs_ids :: Gridap.Arrays.Table
  dof_to_pdof         :: Gridap.Arrays.Table
end

# Issue: I have to pass model, reffe, and conformity, so that I can
#        build the cell_conformity instance. I would have liked to
#        avoid that, given that these were already used in order to
#        build Vh. However, I cannot extract this info out of Vh!!! :-(
function PatchFESpace(space::Gridap.FESpaces.SingleFieldFESpace,
                      patch_decomposition::PatchDecomposition,
                      reffe::Union{ReferenceFE,Tuple{<:Gridap.ReferenceFEs.ReferenceFEName,Any,Any}};
                      conformity=nothing,
                      patches_mask=Fill(false,num_patches(patch_decomposition)))
  cell_conformity = _cell_conformity(patch_decomposition.model,reffe;conformity=conformity)
  return PatchFESpace(space,patch_decomposition,cell_conformity;patches_mask=patches_mask)
end

function PatchFESpace(space::Gridap.FESpaces.SingleFieldFESpace,
                      patch_decomposition::PatchDecomposition,
                      cell_conformity::CellConformity;
                      patches_mask=Fill(false,num_patches(patch_decomposition)))

  cell_dofs_ids = get_cell_dof_ids(space)
  patch_cell_dofs_ids, num_dofs = 
    generate_patch_cell_dofs_ids(get_grid_topology(patch_decomposition.model),
                                 patch_decomposition.patch_cells,
                                 patch_decomposition.patch_cells_overlapped,
                                 patch_decomposition.patch_cells_faces_on_boundary,
                                 cell_dofs_ids,cell_conformity,patches_mask)

  dof_to_pdof = generate_dof_to_pdof(space,patch_decomposition,patch_cell_dofs_ids)
  return PatchFESpace(space,patch_decomposition,num_dofs,patch_cell_dofs_ids,dof_to_pdof)
end

Gridap.FESpaces.get_dof_value_type(a::PatchFESpace)   = Gridap.FESpaces.get_dof_value_type(a.Vh)
Gridap.FESpaces.get_free_dof_ids(a::PatchFESpace)     = Base.OneTo(a.num_dofs)
Gridap.FESpaces.get_fe_basis(a::PatchFESpace)         = get_fe_basis(a.Vh)
Gridap.FESpaces.ConstraintStyle(::PatchFESpace)       = Gridap.FESpaces.UnConstrained()
Gridap.FESpaces.ConstraintStyle(::Type{PatchFESpace}) = Gridap.FESpaces.UnConstrained()
Gridap.FESpaces.get_vector_type(a::PatchFESpace)      = get_vector_type(a.Vh)
Gridap.FESpaces.get_fe_dof_basis(a::PatchFESpace)     = get_fe_dof_basis(a.Vh)

function Gridap.CellData.get_triangulation(a::PatchFESpace)
  PD = a.patch_decomposition
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  trian = get_triangulation(a.Vh)
  return PatchTriangulation(trian,PD,patch_cells,nothing,nothing)
end

# get_cell_dof_ids

Gridap.FESpaces.get_cell_dof_ids(a::PatchFESpace) = a.patch_cell_dofs_ids
Gridap.FESpaces.get_cell_dof_ids(a::PatchFESpace,::Triangulation) = @notimplemented

function Gridap.FESpaces.get_cell_dof_ids(a::PatchFESpace,trian::PatchTriangulation)
  return get_cell_dof_ids(trian.trian,a,trian)
end

function Gridap.FESpaces.get_cell_dof_ids(::Triangulation,a::PatchFESpace,trian::PatchTriangulation)
  return a.patch_cell_dofs_ids
end

function Gridap.FESpaces.get_cell_dof_ids(::BoundaryTriangulation,a::PatchFESpace,trian::PatchTriangulation)
  cell_dof_ids     = get_cell_dof_ids(a)
  pfaces_to_pcells = trian.pfaces_to_pcells
  return lazy_map(Reindex(cell_dof_ids),lazy_map(x->x[1],pfaces_to_pcells))
end

function Gridap.FESpaces.get_cell_dof_ids(::SkeletonTriangulation,a::PatchFESpace,trian::PatchTriangulation)
  cell_dof_ids     = get_cell_dof_ids(a)
  pfaces_to_pcells = trian.pfaces_to_pcells
  
  plus  = lazy_map(Reindex(cell_dof_ids),lazy_map(x->x[1],pfaces_to_pcells))
  minus = lazy_map(Reindex(cell_dof_ids),lazy_map(x->x[2],pfaces_to_pcells))
  return lazy_map(Gridap.Fields.BlockMap(2,[1,2]),plus,minus)
end

# scatter dof values

function Gridap.FESpaces.scatter_free_and_dirichlet_values(f::PatchFESpace,free_values,dirichlet_values)
  cell_vals = Gridap.Fields.PosNegReindex(free_values,dirichlet_values)
  return lazy_map(Broadcasting(cell_vals),f.patch_cell_dofs_ids)
end

# Construction of the patch cell dofs ids

function generate_patch_cell_dofs_ids(topology,
                                      patch_cells,
                                      patch_cells_overlapped,
                                      patch_cells_faces_on_boundary,
                                      cell_dofs_ids,
                                      cell_conformity,
                                      patches_mask)
  patch_cell_dofs_ids = allocate_patch_cell_dofs_ids(patch_cells,cell_dofs_ids)
  num_dofs = generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,topology,
                                           patch_cells,patch_cells_overlapped,
                                           patch_cells_faces_on_boundary,
                                           cell_dofs_ids,cell_conformity,patches_mask)
  return patch_cell_dofs_ids, num_dofs
end

function allocate_patch_cell_dofs_ids(patch_cells,cell_dofs_ids)
  cache_cells = array_cache(patch_cells)
  cache_cdofs = array_cache(cell_dofs_ids)

  num_overlapped_cells = length(patch_cells.data)
  ptrs    = Vector{Int}(undef,num_overlapped_cells+1)
  ptrs[1] = 1; ncells = 1
  for patch = 1:length(patch_cells)
    cells = getindex!(cache_cells,patch_cells,patch)
    for cell in cells
      current_cell_dof_ids = getindex!(cache_cdofs,cell_dofs_ids,cell)
      ptrs[ncells+1] = ptrs[ncells]+length(current_cell_dof_ids)
      ncells += 1
    end
  end

  @check num_overlapped_cells+1 == ncells
  data = Vector{Int}(undef,ptrs[end]-1)
  return Gridap.Arrays.Table(data,ptrs)
end

function generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                       topology,
                                       patch_cells,
                                       patch_cells_overlapped,
                                       patch_cells_faces_on_boundary,
                                       cell_dofs_ids,
                                       cell_conformity,
                                       patches_mask)

    cache = array_cache(patch_cells)
    num_patches = length(patch_cells)
    current_dof = 1
    for patch = 1:num_patches
      current_patch_cells = getindex!(cache,patch_cells,patch)
      current_dof = generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                    topology,
                                    patch,
                                    current_patch_cells,
                                    patch_cells_overlapped,
                                    patch_cells_faces_on_boundary,
                                    cell_dofs_ids,
                                    cell_conformity;
                                    free_dofs_offset=current_dof,
                                    mask=patches_mask[patch])
    end
    return current_dof-1
end

# TO-THINK/STRESS:
#  1. MultiFieldFESpace case?
#  2. FESpaces which are directly defined on physical space? We think this case is covered by
#     the fact that we are using a CellConformity instance to rely on ownership info.
# free_dofs_offset     : the ID from which we start to assign free DoF IDs upwards
# Note: we do not actually need to generate a global numbering for Dirichlet DoFs. We can
#       tag all as them with -1, as we are always imposing homogenous Dirichlet boundary
#       conditions, and thus there is no need to address the result of interpolating Dirichlet
#       Data into the FE space.
function generate_patch_cell_dofs_ids!(patch_cell_dofs_ids,
                                       topology,
                                       patch::Integer,
                                       patch_cells::AbstractVector{<:Integer},
                                       patch_cells_overlapped::Gridap.Arrays.Table,
                                       patch_cells_faces_on_boundary,
                                       global_space_cell_dofs_ids,
                                       cell_conformity;
                                       free_dofs_offset=1,
                                       mask=false)

  o  = patch_cells_overlapped.ptrs[patch]
  if mask
    for lpatch_cell = 1:length(patch_cells)
      cell_overlapped_mesh = patch_cells_overlapped.data[o+lpatch_cell-1]
      s = patch_cell_dofs_ids.ptrs[cell_overlapped_mesh]
      e = patch_cell_dofs_ids.ptrs[cell_overlapped_mesh+1]-1
      patch_cell_dofs_ids.data[s:e] .= -1
    end
  else
    g2l = Dict{Int,Int}()
    Dc  = length(patch_cells_faces_on_boundary)
    d_to_cell_to_dface = [Gridap.Geometry.get_faces(topology,Dc,d) for d in 0:Dc-1]

    # Loop over cells of the patch (local_cell_id_within_patch)
    for (lpatch_cell,patch_cell) in enumerate(patch_cells)
      cell_overlapped_mesh = patch_cells_overlapped.data[o+lpatch_cell-1]
      s = patch_cell_dofs_ids.ptrs[cell_overlapped_mesh]
      e = patch_cell_dofs_ids.ptrs[cell_overlapped_mesh+1]-1
      current_patch_cell_dofs_ids = view(patch_cell_dofs_ids.data,s:e)
      ctype = cell_conformity.cell_ctype[patch_cell]

      # 1) DoFs belonging to faces (Df < Dc)
      face_offset = 0
      for d = 0:Dc-1
        num_cell_faces = length(d_to_cell_to_dface[d+1][patch_cell])
        for lface in 1:num_cell_faces
          for ldof in cell_conformity.ctype_lface_own_ldofs[ctype][face_offset+lface]
            gdof = global_space_cell_dofs_ids[patch_cell][ldof]
            
            face_in_patch_boundary = patch_cells_faces_on_boundary[d+1][cell_overlapped_mesh][lface]
            dof_is_dirichlet = (gdof < 0)
            if face_in_patch_boundary || dof_is_dirichlet
              current_patch_cell_dofs_ids[ldof] = -1
            elseif gdof in keys(g2l)
              current_patch_cell_dofs_ids[ldof] = g2l[gdof]
            else
              g2l[gdof] = free_dofs_offset
              current_patch_cell_dofs_ids[ldof] = free_dofs_offset
              free_dofs_offset += 1
            end
          end
        end
        face_offset += cell_conformity.d_ctype_num_dfaces[d+1][ctype]
      end

      # 2) Interior DoFs
      for ldof in cell_conformity.ctype_lface_own_ldofs[ctype][face_offset+1]
        current_patch_cell_dofs_ids[ldof] = free_dofs_offset
        free_dofs_offset += 1
      end
    end
  end
  return free_dofs_offset
end

function generate_dof_to_pdof(Vh,PD,patch_cell_dofs_ids)
  dof_to_pdof = _allocate_dof_to_pdof(Vh,PD,patch_cell_dofs_ids)
  _generate_dof_to_pdof!(dof_to_pdof,Vh,PD,patch_cell_dofs_ids)
  return dof_to_pdof
end

function _allocate_dof_to_pdof(Vh,PD,patch_cell_dofs_ids)
  touched = Dict{Int,Bool}()
  cell_mesh_overlapped = 1
  cache_patch_cells  = array_cache(PD.patch_cells)
  cell_dof_ids       = get_cell_dof_ids(Vh)
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  ptrs = fill(0,num_free_dofs(Vh)+1)
  for patch = 1:length(PD.patch_cells)
    current_patch_cells = getindex!(cache_patch_cells,PD.patch_cells,patch)
    for cell in current_patch_cells
      current_cell_dof_ids = getindex!(cache_cell_dof_ids,cell_dof_ids,cell)
      s = patch_cell_dofs_ids.ptrs[cell_mesh_overlapped]
      e = patch_cell_dofs_ids.ptrs[cell_mesh_overlapped+1]-1
      current_patch_cell_dof_ids = view(patch_cell_dofs_ids.data,s:e)
      for (dof,pdof) in zip(current_cell_dof_ids,current_patch_cell_dof_ids)
        if pdof > 0 && !(dof ∈ keys(touched))
          touched[dof] = true
          ptrs[dof+1] += 1
        end
      end
      cell_mesh_overlapped += 1
    end
    empty!(touched)
  end
  PartitionedArrays.length_to_ptrs!(ptrs)

  data = fill(0,ptrs[end]-1)
  return Gridap.Arrays.Table(data,ptrs)
end

function _generate_dof_to_pdof!(dof_to_pdof,Vh,PD,patch_cell_dofs_ids)
  touched = Dict{Int,Bool}()
  cell_mesh_overlapped = 1
  cache_patch_cells  = array_cache(PD.patch_cells)
  cell_dof_ids       = get_cell_dof_ids(Vh)
  cache_cell_dof_ids = array_cache(cell_dof_ids)

  ptrs = dof_to_pdof.ptrs
  data = dof_to_pdof.data
  local_ptrs = fill(Int32(0),num_free_dofs(Vh))
  for patch = 1:length(PD.patch_cells)
    current_patch_cells = getindex!(cache_patch_cells,PD.patch_cells,patch)
    for cell in current_patch_cells
      current_cell_dof_ids = getindex!(cache_cell_dof_ids,cell_dof_ids,cell)
      s = patch_cell_dofs_ids.ptrs[cell_mesh_overlapped]
      e = patch_cell_dofs_ids.ptrs[cell_mesh_overlapped+1]-1
      current_patch_cell_dof_ids = view(patch_cell_dofs_ids.data,s:e)
      for (dof,pdof) in zip(current_cell_dof_ids,current_patch_cell_dof_ids)
        if pdof > 0 && !(dof ∈ keys(touched))
          touched[dof] = true
          idx = ptrs[dof] + local_ptrs[dof]
          @check idx < ptrs[dof+1]
          data[idx] = pdof
          local_ptrs[dof] += 1
        end
      end
      cell_mesh_overlapped += 1
    end
    empty!(touched)
  end
end

# x \in  PatchFESpace
# y \in  SingleFESpace
function prolongate!(x,Ph::PatchFESpace,y;dof_ids=LinearIndices(y))
  dof_to_pdof = Ph.dof_to_pdof
  
  ptrs = dof_to_pdof.ptrs
  data = dof_to_pdof.data
  for dof in dof_ids
    for k in ptrs[dof]:ptrs[dof+1]-1
      pdof = data[k]
      x[pdof] = y[dof]
    end
  end
end

# x \in  SingleFESpace
# y \in  PatchFESpace
function inject!(x,Ph::PatchFESpace,y)
  dof_to_pdof = Ph.dof_to_pdof
  
  ptrs = dof_to_pdof.ptrs
  data = dof_to_pdof.data
  for dof in 1:length(dof_to_pdof)
    x[dof] = 0.0
    for k in ptrs[dof]:ptrs[dof+1]-1
      pdof = data[k]
      x[dof] += y[pdof]
    end
  end
end

function inject!(x,Ph::PatchFESpace,y,w,w_sums)
  dof_to_pdof = Ph.dof_to_pdof
  
  ptrs = dof_to_pdof.ptrs
  data = dof_to_pdof.data
  for dof in 1:length(dof_to_pdof)
    x[dof] = 0.0
    for k in ptrs[dof]:ptrs[dof+1]-1
      pdof = data[k]
      x[dof] += y[pdof] * w[pdof]
    end
    x[dof] /= w_sums[dof]
  end
end

function compute_weight_operators(Ph::PatchFESpace,Vh)
  w      = Fill(1.0,num_free_dofs(Ph))
  w_sums = zeros(num_free_dofs(Vh))
  inject!(w_sums,Ph,w,Fill(1.0,num_free_dofs(Ph)),Fill(1.0,num_free_dofs(Vh)))
  return w, w_sums
end