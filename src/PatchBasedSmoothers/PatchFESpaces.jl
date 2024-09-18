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

"""
    struct PatchFESpace <: SingleFieldFESpace
      ...
    end

FESpace representing a patch-based subspace decomposition `V = Σ_i V_i` of a global space `V`.
"""
struct PatchFESpace{A,B} <: FESpaces.SingleFieldFESpace
  Vh                  :: A
  patch_decomposition :: PatchDecomposition
  num_dofs            :: Int
  patch_cell_dofs_ids :: Arrays.Table
  dof_to_pdof         :: Arrays.Table
  metadata            :: B

  function PatchFESpace(
    space::SingleFieldFESpace,
    patch_decomposition::PatchDecomposition,
    num_dofs,patch_cell_dofs_ids,dof_to_pdof,
    matadata=nothing
  )
    A = typeof(space)
    B = typeof(matadata)
    new{A,B}(space,patch_decomposition,num_dofs,patch_cell_dofs_ids,dof_to_pdof,matadata)
  end
end

@doc """
    function PatchFESpace(
      space::FESpaces.SingleFieldFESpace,
      patch_decomposition::PatchDecomposition,
      reffe::Union{ReferenceFE,Tuple{<:ReferenceFEs.ReferenceFEName,Any,Any}};
      conformity=nothing,
      patches_mask=Fill(false,num_patches(patch_decomposition))
    )

Constructs a `PatchFESpace` from a global `SingleFieldFESpace` and a `PatchDecomposition`.
The conformity of the FESpace is deduced from `reffe` and `conformity`, which need to be 
the same as the ones used to construct the global FESpace.

If `patches_mask[p] = true`, the patch `p` is ignored. Used in parallel.
"""
function PatchFESpace(
  space::FESpaces.SingleFieldFESpace,
  patch_decomposition::PatchDecomposition,
  reffe::Union{ReferenceFE,Tuple{<:ReferenceFEs.ReferenceFEName,Any,Any}};
  conformity=nothing,
  patches_mask=Fill(false,num_patches(patch_decomposition))
)
  cell_conformity = MultilevelTools._cell_conformity(patch_decomposition.model,reffe;conformity=conformity)
  return PatchFESpace(space,patch_decomposition,cell_conformity;patches_mask=patches_mask)
end

@doc """
    function PatchFESpace(
      space::FESpaces.SingleFieldFESpace,
      patch_decomposition::PatchDecomposition,
      cell_conformity::CellConformity;
      patches_mask=Fill(false,num_patches(patch_decomposition))
    )

Constructs a `PatchFESpace` from a global `SingleFieldFESpace`, a `PatchDecomposition`
and a `CellConformity` instance.

If `patches_mask[p] = true`, the patch `p` is ignored. Used in parallel.
"""
function PatchFESpace(
  space::FESpaces.SingleFieldFESpace,
  patch_decomposition::PatchDecomposition,
  cell_conformity::CellConformity;
  patches_mask = Fill(false,num_patches(patch_decomposition))
)
  cell_dofs_ids = get_cell_dof_ids(space)
  patch_cells_overlapped = get_patch_cells_overlapped(patch_decomposition)
  patch_cell_dofs_ids, num_dofs = generate_patch_cell_dofs_ids(
    patch_decomposition.patch_cells,
    patch_cells_overlapped,
    patch_decomposition.patch_cells_faces_on_boundary,
    cell_dofs_ids,cell_conformity;patches_mask
  )
  dof_to_pdof = generate_dof_to_pdof(space,patch_decomposition,patch_cell_dofs_ids)
  return PatchFESpace(space,patch_decomposition,num_dofs,patch_cell_dofs_ids,dof_to_pdof)
end

FESpaces.get_dof_value_type(a::PatchFESpace)     = Gridap.FESpaces.get_dof_value_type(a.Vh)
FESpaces.get_free_dof_ids(a::PatchFESpace)       = Base.OneTo(a.num_dofs)
FESpaces.get_fe_basis(a::PatchFESpace)           = get_fe_basis(a.Vh)
FESpaces.ConstraintStyle(::PatchFESpace)         = Gridap.FESpaces.UnConstrained()
FESpaces.ConstraintStyle(::Type{<:PatchFESpace}) = Gridap.FESpaces.UnConstrained()
FESpaces.get_vector_type(a::PatchFESpace)        = get_vector_type(a.Vh)
FESpaces.get_fe_dof_basis(a::PatchFESpace)       = get_fe_dof_basis(a.Vh)

function Gridap.CellData.get_triangulation(a::PatchFESpace)
  PD = a.patch_decomposition
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  trian = get_triangulation(a.Vh)
  return PatchTriangulation(trian,PD,patch_cells,nothing)
end

# get_cell_dof_ids

FESpaces.get_cell_dof_ids(a::PatchFESpace) = a.patch_cell_dofs_ids
FESpaces.get_cell_dof_ids(a::PatchFESpace,::Triangulation) = @notimplemented

function FESpaces.get_cell_dof_ids(a::PatchFESpace,trian::PatchTriangulation)
  return get_cell_dof_ids(trian.trian,a,trian)
end

function FESpaces.get_cell_dof_ids(t::Gridap.Adaptivity.AdaptedTriangulation,a::PatchFESpace,trian::PatchTriangulation)
  return get_cell_dof_ids(t.trian,a,trian)
end

function FESpaces.get_cell_dof_ids(::Triangulation,a::PatchFESpace,trian::PatchTriangulation)
  return a.patch_cell_dofs_ids
end

function FESpaces.get_cell_dof_ids(::BoundaryTriangulation,a::PatchFESpace,trian::PatchTriangulation)
  cell_dof_ids     = get_cell_dof_ids(a)
  pface_to_pcell = trian.pface_to_pcell
  pcells = isempty(pface_to_pcell) ? Int[] : lazy_map(x->x[1],pface_to_pcell)
  return lazy_map(Reindex(cell_dof_ids),pcells)
end

function FESpaces.get_cell_dof_ids(::SkeletonTriangulation,a::PatchFESpace,trian::PatchTriangulation)
  cell_dof_ids     = get_cell_dof_ids(a)
  pface_to_pcell = trian.pface_to_pcell

  pcells_plus  = isempty(pface_to_pcell) ? Int[] : lazy_map(x->x[1],pface_to_pcell)
  pcells_minus = isempty(pface_to_pcell) ? Int[] : lazy_map(x->x[2],pface_to_pcell)
  
  plus  = lazy_map(Reindex(cell_dof_ids),pcells_plus)
  minus = lazy_map(Reindex(cell_dof_ids),pcells_minus)
  return lazy_map(Fields.BlockMap(2,[1,2]),plus,minus)
end

function FESpaces.get_cell_dof_ids(a::PatchFESpace,trian::PatchClosureTriangulation)
  patch_cells = trian.trian.patch_faces
  return propagate_patch_dof_ids(a,patch_cells)
end

# scatter dof values

function FESpaces.scatter_free_and_dirichlet_values(f::PatchFESpace,free_values,dirichlet_values)
  cell_vals = Fields.PosNegReindex(free_values,dirichlet_values)
  return lazy_map(Broadcasting(cell_vals),f.patch_cell_dofs_ids)
end

# Construction of the patch cell dofs ids

function generate_patch_cell_dofs_ids(
  patch_cells,
  patch_cells_overlapped,
  patch_cells_faces_on_boundary,
  cell_dofs_ids,
  cell_conformity::CellConformity;
  patches_mask = Fill(false,length(patch_cells)),
  numbering = :global
)
  @assert numbering in [:global,:local]
  patch_cell_dofs_ids = allocate_patch_cell_array(patch_cells,cell_dofs_ids;init=Int32(-1))
  num_dofs = generate_patch_cell_dofs_ids!(
    patch_cell_dofs_ids,
    patch_cells,patch_cells_overlapped,
    patch_cells_faces_on_boundary,
    cell_dofs_ids,cell_conformity;
    patches_mask,numbering
  )
  return patch_cell_dofs_ids, num_dofs
end

function generate_patch_cell_dofs_ids!(
  patch_cell_dofs_ids,
  patch_cells,
  patch_cells_overlapped,
  patch_cells_faces_on_boundary,
  cell_dofs_ids,
  cell_conformity::CellConformity;
  patches_mask = Fill(false,length(patch_cells)),
  numbering = :global
)
  @assert numbering in [:global,:local]
  dof_offset = 1
  for patch = 1:length(patch_cells)
    current_patch_cells = view(patch_cells,patch)
    dof_offset = generate_patch_cell_dofs_ids!(
      patch_cell_dofs_ids,
      patch,current_patch_cells,patch_cells_overlapped,
      patch_cells_faces_on_boundary,
      cell_dofs_ids,cell_conformity;
      dof_offset=dof_offset,
      mask=patches_mask[patch]
    )
    if numbering == :local
      dof_offset = 1
    end
  end
  return dof_offset-1
end

# Note: We do not actually need to generate a global numbering for Dirichlet DoFs. We can
# tag all as them with -1, as we are always imposing homogenous Dirichlet boundary
# conditions, and thus there is no need to address the result of interpolating Dirichlet
# Data into the FE space.
function generate_patch_cell_dofs_ids!(
  patch_cell_dofs_ids,
  patch::Integer,
  patch_cells::AbstractVector{<:Integer},
  patch_cells_overlapped::Gridap.Arrays.Table,
  patch_cells_faces_on_boundary,
  global_space_cell_dofs_ids,
  cell_conformity::CellConformity;
  dof_offset=1,
  mask=false
)
  o = patch_cells_overlapped.ptrs[patch]
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

    # Loop over cells of the patch (local_cell_id_within_patch)
    for (lpatch_cell,patch_cell) in enumerate(patch_cells)
      cell_overlapped_mesh = patch_cells_overlapped.data[o+lpatch_cell-1]
      s = patch_cell_dofs_ids.ptrs[cell_overlapped_mesh]
      e = patch_cell_dofs_ids.ptrs[cell_overlapped_mesh+1]-1
      current_patch_cell_dofs_ids = view(patch_cell_dofs_ids.data,s:e)
      ctype = cell_conformity.cell_ctype[patch_cell]

      face_offset = 0
      for d = 0:Dc
        n_dfaces = cell_conformity.d_ctype_num_dfaces[d+1][ctype]
        for lface in 1:n_dfaces
          for ldof in cell_conformity.ctype_lface_own_ldofs[ctype][face_offset+lface]
            gdof = global_space_cell_dofs_ids[patch_cell][ldof]
            
            face_in_patch_boundary = (d != Dc) && patch_cells_faces_on_boundary[d+1][cell_overlapped_mesh][lface]
            dof_is_dirichlet = (gdof < 0)
            if face_in_patch_boundary || dof_is_dirichlet
              current_patch_cell_dofs_ids[ldof] = -1
            elseif gdof in keys(g2l)
              current_patch_cell_dofs_ids[ldof] = g2l[gdof]
            else
              g2l[gdof] = dof_offset
              current_patch_cell_dofs_ids[ldof] = dof_offset
              dof_offset += 1
            end
          end
        end
        face_offset += cell_conformity.d_ctype_num_dfaces[d+1][ctype]
      end
    end
  end
  return dof_offset
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

function generate_pdof_to_dof(
  patch_decomposition::PatchDecomposition,
  cell_dof_ids::Table{Ti},
  patch_cell_lids::Table{Ti}
) where Ti <: Integer

  n_patches = num_patches(patch_decomposition)
  patch_cells = get_patch_cells(patch_decomposition)

  ptrs = fill(0,n_patches+1)
  for patch in 1:n_patches
    cell_lids = patch_view(patch_decomposition,patch_cell_lids,patch)
    ptrs[patch+1] = maximum(map(maximum,cell_lids))
  end
  PartitionedArrays.length_to_ptrs!(ptrs)

  data = fill(0,ptrs[end]-1)
  for (patch,cells) in enumerate(patch_cells)
    cell_lids = patch_view(patch_decomposition,patch_cell_lids,patch)
    for (lcell,cell) in enumerate(cells)
      dofs = view(cell_dof_ids,cell)
      pdofs = view(cell_lids,lcell)
      for (ldof,dof) in zip(pdofs,dofs)
        if ldof > 0
          data[ptrs[patch]+ldof-1] = dof
        end
      end
    end
  end

  return Arrays.Table(data,ptrs)
end

# TODO: Just For lagrange multipliers, fiz this better
function generate_pdof_to_dof(
  patch_decomposition::PatchDecomposition,
  cell_dof_ids::Fill,
  patch_cell_lids::Table{Ti}
) where Ti <: Integer
  ptrs = collect(1:num_patches(patch_decomposition)+1)
  data = collect(Fill(1,num_patches(patch_decomposition)))
  return Arrays.Table(data,ptrs)
end

"""
    propagate_patch_dof_ids(patch_space::PatchFESpace,new_patch_cells::Table)

Propagates the DoF ids of the patch_space to a new set of patch cells given by 
a patch-wise Table `new_patch_cells`.
"""
function propagate_patch_dof_ids(
  patch_space::PatchFESpace,
  new_patch_cells::Table
)
  space = patch_space.Vh
  patch_decomposition = patch_space.patch_decomposition
  patch_cells = get_patch_cells(patch_decomposition)
  patch_pcells = get_patch_cells_overlapped(patch_decomposition)

  cell_dof_ids = get_cell_dof_ids(space)
  patch_cell_dof_ids = patch_space.patch_cell_dofs_ids
  ext_dof_ids = allocate_patch_cell_array(new_patch_cells,cell_dof_ids;init=Int32(-1))

  new_pcell = 1
  n_patches = length(patch_cells)
  for patch in 1:n_patches
    cells = view(patch_cells,patch)
    pcells = view(patch_pcells,patch)

    # Create local dof to pdof maps
    dof_to_pdof = Dict{Int,Int}()
    for (cell,pcell) in zip(cells,pcells)
      dofs = view(cell_dof_ids,cell)
      pdofs = view(patch_cell_dof_ids,pcell)
      for (dof,pdof) in zip(dofs,pdofs)
        if pdof != -1
          dof_to_pdof[dof] = pdof
        end
      end
    end

    # Propagate dofs to patch extensions
    for new_cell in view(new_patch_cells,patch)
      dofs = view(cell_dof_ids,new_cell)
      pdofs = view(ext_dof_ids,new_pcell)

      pos = findfirst(c -> c == new_cell, cells)
      if !isnothing(pos) # Cell is in the patch
        pcell = pcells[pos]
        pdofs .= view(patch_cell_dof_ids,pcell)
      else # Cell is new (not in the patch)
        for (ldof,dof) in enumerate(dofs)
          if haskey(dof_to_pdof,dof)
            pdofs[ldof] = dof_to_pdof[dof]
          else
            pdofs[ldof] = -1
          end
        end
      end
      new_pcell += 1
    end
  end

  return ext_dof_ids
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