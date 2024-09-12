
"""
    abstract type PatchBoundaryStyle end
    struct PatchBoundaryExclude  <: PatchBoundaryStyle end
    struct PatchBoundaryInclude  <: PatchBoundaryStyle end
  
Controls the boundary consitions imposed at the patch boundaries for the sub-spaces.
- `PatchBoundaryInclude`: No BCs are imposed at the patch boundaries. 
- `PatchBoundaryExclude`: Zero dirichlet BCs are imposed at the patch boundaries.
"""
abstract type PatchBoundaryStyle end
struct PatchBoundaryExclude  <: PatchBoundaryStyle end
struct PatchBoundaryInclude  <: PatchBoundaryStyle end

"""
    struct PatchDecomposition{Dr,Dc,Dp} <: DiscreteModel{Dc,Dp}

Represents a patch decomposition of a discrete model, i.e an overlapping cell covering `{Ω_i}`
of `Ω` such that `Ω = Σ_i Ω_i`.

## Properties: 

- `Dr::Integer` : Dimension of the patch root
- `model::DiscreteModel{Dc,Dp}` : Underlying discrete model
- `patch_cells::Table` : [patch][local cell] -> cell
- `patch_cells_faces_on_boundary::Table` : [d][overlapped cell][local face] -> face is on patch boundary

"""
struct PatchDecomposition{Dr,Dc,Dp} <: GridapType
  model                         :: DiscreteModel{Dc,Dp}
  patch_cells                   :: Arrays.Table         # [patch][local cell] -> cell
  patch_cells_faces_on_boundary :: Vector{Arrays.Table} # [d][overlapped cell][local face] -> face is on patch boundary
  patch_boundary_style          :: PatchBoundaryStyle
end

"""
    function PatchDecomposition(
      model::DiscreteModel{Dc,Dp};
      Dr=0,
      patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude(),
      boundary_tag_names::AbstractArray{String}=["boundary"]
    )

Returns an instance of [`PatchDecomposition`](@ref) from a given discrete model.
"""
function PatchDecomposition(
  model::DiscreteModel{Dc,Dp};
  Dr::Integer=0,
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude(),
  boundary_tag_names::AbstractArray{String}=["boundary"]
) where {Dc,Dp}
  @assert 0 <= Dr <= Dc-1

  topology     = get_grid_topology(model)
  patch_cells  = Geometry.get_faces(topology,Dr,Dc)
  patch_facets = Geometry.get_faces(topology,Dr,Dc-1)

  patch_cells_faces_on_boundary = compute_patch_cells_faces_on_boundary(
    model, patch_cells, patch_facets, patch_boundary_style, boundary_tag_names
  )

  return PatchDecomposition{Dr,Dc,Dp}(
    model, patch_cells, patch_cells_faces_on_boundary, patch_boundary_style
  )
end

"""
    num_patches(a::PatchDecomposition)
"""
num_patches(a::PatchDecomposition) = length(a.patch_cells)

"""
    get_patch_cells(PD::PatchDecomposition) -> patch_to_cells
"""
get_patch_cells(PD::PatchDecomposition) = PD.patch_cells

"""
    get_patch_cell_offsets(PD::PatchDecomposition)
"""
get_patch_cell_offsets(PD::PatchDecomposition) = PD.patch_cells.ptrs

Geometry.num_cells(a::PatchDecomposition) = length(a.patch_cells.data)
Geometry.get_isboundary_face(PD::PatchDecomposition) = PD.patch_cells_faces_on_boundary
Geometry.get_isboundary_face(PD::PatchDecomposition,d::Integer) = PD.patch_cells_faces_on_boundary[d+1]

"""
    get_patch_cells_overlapped(PD::PatchDecomposition) -> patch_to_pcells
"""
function get_patch_cells_overlapped(PD::PatchDecomposition)
  patch_cells = get_patch_cells(PD)
  n_pcells = length(patch_cells.data)
  data = Arrays.IdentityVector(n_pcells)
  return Arrays.Table(data,patch_cells.ptrs)
end

function Base.view(PD::PatchDecomposition{Dr,Dc,Dp},patch_ids) where {Dr,Dc,Dp}
  patch_cells = view(get_patch_cells(PD),patch_ids)
  patch_faces_on_boundary = [
    view(PD.patch_cells_faces_on_boundary[d+1],patch_ids) for d = 0:Dc-1
  ]
  return PatchDecomposition{Dr,Dc,Dp}(PD.model,patch_cells,patch_faces_on_boundary,PD.patch_boundary_style)
end

"""
    patch_view(PD::PatchDecomposition,a::AbstractArray,patch::Integer)
    patch_view(PD::PatchDecomposition,a::AbstractArray,patch_ids::AbstractUnitRange{<:Integer})

Returns a view of the pcell-wise array `a` restricted to the pcells of the patch `patch` or `patch_ids`.
"""
function patch_view(PD::PatchDecomposition,a::AbstractArray,patch::Integer)
  offsets = get_patch_cell_offsets(PD)
  return view(a,offsets[patch]:offsets[patch+1]-1)
end

function patch_view(PD::PatchDecomposition,a::AbstractArray,patch_ids::AbstractUnitRange{<:Integer})
  offsets = get_patch_cell_offsets(PD)
  start = offsets[first(patch_ids)]
  stop  = offsets[last(patch_ids)+1]-1
  return view(a,start:stop)
end

"""
    patch_reindex(PD::PatchDecomposition,cell_to_data) -> pcell_to_data
"""
function patch_reindex(PD::PatchDecomposition,cell_to_data)
  patch_cells = get_patch_cells(PD)
  pcell_to_data = lazy_map(Reindex(cell_to_data),patch_cells.data)
  return pcell_to_data
end

"""
    allocate_patch_cell_array(PD::PatchDecomposition,cell_to_data::Table{T};init=zero(T))
  
Allocates a patch-cell-wise array from a cell-wise array.
"""
function allocate_patch_cell_array(
  PD::PatchDecomposition, cell_to_data::Table{T}; init=zero(T)
) where T
  patch_cells = get_patch_cells(PD)
  return allocate_patch_cell_array(patch_cells,cell_to_data;init)
end

function allocate_patch_cell_array(
  patch_cells::Table, cell_to_data::Table{T}; init = zero(T)
) where T
  ptrs = zeros(Int,length(patch_cells.data)+1)
  ptrs[1] = 1
  for (pcell,cell) in enumerate(patch_cells.data)
    n = cell_to_data.ptrs[cell+1] - cell_to_data.ptrs[cell]
    ptrs[pcell+1] = ptrs[pcell] + n
  end
  data = fill(init,ptrs[end]-1)
  return Arrays.Table(data,ptrs)
end

function allocate_patch_cell_array(
  patch_cells::Table, cell_to_data::AbstractVector{<:AbstractVector{T}}; init = zero(T)
) where T
  ptrs = zeros(Int,length(patch_cells.data)+1)
  ptrs[1] = 1
  for (pcell,cell) in enumerate(patch_cells.data)
    n = length(cell_to_data[cell])
    ptrs[pcell+1] = ptrs[pcell] + n
  end
  data = fill(init,ptrs[end]-1)
  return Arrays.Table(data,ptrs)
end

# patch_cell_faces_on_boundary :: 
#    [Df][overlapped cell][lface] -> Face is boundary of the patch
function compute_patch_cells_faces_on_boundary(
  model::DiscreteModel,
  patch_cells,
  patch_facets,
  patch_boundary_style,
  boundary_tag_names
)
  patch_cell_faces_on_boundary = _allocate_patch_cells_faces_on_boundary(model,patch_cells)
  if isa(patch_boundary_style,PatchBoundaryExclude)
    _compute_patch_cells_faces_on_boundary!(
      patch_cell_faces_on_boundary,
      model, patch_cells,patch_facets, 
      patch_boundary_style, boundary_tag_names
    )
  end
  return patch_cell_faces_on_boundary
end

function _allocate_patch_cells_faces_on_boundary(model::DiscreteModel{Dc},patch_cells) where {Dc}
  ctype_to_reffe = get_reffes(model)
  cell_to_ctype  = get_cell_type(model)
  
  patch_cells_faces_on_boundary = Vector{Gridap.Arrays.Table}(undef,Dc)
  for d = 0:Dc-1
    ctype_to_num_dfaces = map(reffe -> num_faces(reffe,d), ctype_to_reffe)
    patch_cells_faces_on_boundary[d+1] =
      _allocate_ocell_to_dface(Bool, patch_cells,cell_to_ctype, ctype_to_num_dfaces)
  end
  return patch_cells_faces_on_boundary
end

function _allocate_ocell_to_dface(::Type{T},patch_cells,cell_to_ctype,ctype_to_num_dfaces) where T<:Number
  n_pcells = length(patch_cells.data)
  ptrs = Vector{Int}(undef,n_pcells+1)

  ptrs[1] = 1
  for i = 1:n_pcells
    cell  = patch_cells.data[i]
    ctype = cell_to_ctype[cell]
    ptrs[i+1] = ptrs[i] + ctype_to_num_dfaces[ctype]
  end
  data = zeros(T,ptrs[end]-1)
  return Gridap.Arrays.Table(data,ptrs)
end

function _compute_patch_cells_faces_on_boundary!(
  patch_cells_faces_on_boundary,
  model::DiscreteModel,
  patch_cells,
  patch_facets,
  patch_boundary_style,
  boundary_tag_names
)
  num_patches = length(patch_cells.ptrs)-1
  cache_patch_cells  = array_cache(patch_cells)
  cache_patch_facets = array_cache(patch_facets)
  for patch = 1:num_patches
    first_patch_cell = patch_cells.ptrs[patch]
    current_patch_cells  = getindex!(cache_patch_cells,patch_cells,patch)
    current_patch_facets = getindex!(cache_patch_facets,patch_facets,patch)
    _compute_patch_cells_faces_on_boundary!(
      patch_cells_faces_on_boundary,
      model,
      first_patch_cell,
      current_patch_cells,
      current_patch_facets,
      patch_boundary_style,
      boundary_tag_names
    )
  end
end

function _compute_patch_cells_faces_on_boundary!(
  patch_cells_faces_on_boundary,
  model::DiscreteModel{Dc},
  first_patch_cell,
  patch_cells,
  patch_facets,
  patch_boundary_style,
  boundary_tag_names
) where Dc
  face_labeling = get_face_labeling(model)
  topology = get_grid_topology(model)

  boundary_tags = findall(x -> (x ∈ boundary_tag_names), face_labeling.tag_to_name)
  @assert !isempty(boundary_tags)
  boundary_entities = vcat(face_labeling.tag_to_entities[boundary_tags]...)

  # Cells facets
  Df = Dc-1
  cell_to_facets = Gridap.Geometry.get_faces(topology,Dc,Df)
  cache_cell_to_facets = array_cache(cell_to_facets)
  facet_to_cells = Gridap.Geometry.get_faces(topology,Df,Dc)
  cache_facet_to_cells = array_cache(facet_to_cells)

  d_to_facet_to_dfaces = [Gridap.Geometry.get_faces(topology,Df,d) for d = 0:Df-1]
  d_to_cell_to_dfaces  = [Gridap.Geometry.get_faces(topology,Dc,d) for d = 0:Df-1]
  d_to_dface_to_cells  = [Gridap.Geometry.get_faces(topology,d,Dc) for d = 0:Df-1]

  # Go over all cells in the current patch
  for (lcell,cell) in enumerate(patch_cells)
    overlapped_cell = first_patch_cell+lcell-1
    cell_facets = getindex!(cache_cell_to_facets,cell_to_facets,cell)
    # Go over the facets (i.e., faces of dim Dc-1) in the current cell
    for (lfacet,facet) in enumerate(cell_facets)
      facet_entity = face_labeling.d_to_dface_to_entity[Df+1][facet]
      cells_around_facet = getindex!(cache_facet_to_cells,facet_to_cells,facet)

      # Check if facet has a neighboring cell that does not belong to the patch
      has_nbor_outside_patch = false
      for c in cells_around_facet
        if c ∉ patch_cells
          has_nbor_outside_patch = true
          break
        end
      end
      facet_at_global_boundary = (facet_entity ∈ boundary_entities) && (facet ∉ patch_facets)
      facet_at_patch_boundary  = facet_at_global_boundary || has_nbor_outside_patch

      if (facet_at_patch_boundary)
        # Mark the facet as boundary
        position = patch_cells_faces_on_boundary[Df+1].ptrs[overlapped_cell]+lfacet-1
        patch_cells_faces_on_boundary[Df+1].data[position] = true

        # Go over the faces of lower dimension on the boundary of the current facet, 
        # and mark them as boundary as well. 
        for d = 0:Df-1
          for facet_face in d_to_facet_to_dfaces[d+1][facet]
            # Locate the local position of the face within the cell (lface)
            for cell_around_face in d_to_dface_to_cells[d+1][facet_face]
              if cell_around_face ∈ patch_cells
                cell_dfaces = d_to_cell_to_dfaces[d+1][cell_around_face]
                lface       = findfirst(x -> x==facet_face, cell_dfaces)
                lcell2      = findfirst(x -> x==cell_around_face, patch_cells)

                overlapped_cell2 = first_patch_cell+lcell2-1
                position = patch_cells_faces_on_boundary[d+1].ptrs[overlapped_cell2]+lface-1
                patch_cells_faces_on_boundary[d+1].data[position] = true
              end
            end
          end
        end
      end
    end
  end
end

"""
    get_patch_cell_faces(PD::PatchDecomposition,Df::Integer)
    get_patch_cell_faces(PD::PatchDecomposition,Df::Integer,faces_mask::AbstractVector{Bool})

Returns a patch-wise Table containing the faces on each patch cell, i.e 

    patch_faces[pcell] = [face1, face2, ...]

where face1, face2, ... are the faces on the overlapped cell `pcell` such that 

  - they are NOT on the boundary of the patch
  - they are flagged `true` in `faces_mask`
"""
function get_patch_cell_faces(PD::PatchDecomposition,Df::Integer)
  model    = PD.model
  topo     = get_grid_topology(model)
  faces_mask = Fill(true,num_faces(topo,Df))
  return get_patch_cell_faces(PD,Df,faces_mask)
end

function get_patch_cell_faces(
  PD::PatchDecomposition{Dr,Dc},Df::Integer,faces_mask::AbstractVector{Bool}
) where {Dr,Dc}
  model    = PD.model
  topo     = get_grid_topology(model)

  c2e_map  = Gridap.Geometry.get_faces(topo,Dc,Df)
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  patch_cell_faces  = lazy_map(Reindex(c2e_map),patch_cells.data)
  faces_on_boundary = PD.patch_cells_faces_on_boundary[Df+1]

  patch_faces = _allocate_patch_cell_faces(patch_cell_faces,faces_on_boundary,faces_mask)
  _generate_patch_cell_faces!(patch_faces,patch_cell_faces,faces_on_boundary,faces_mask)
  return patch_faces
end

function _allocate_patch_cell_faces(patch_cell_faces,faces_on_boundary,faces_mask)
  num_patch_cells = length(patch_cell_faces)

  num_patch_faces = 0
  patch_cells_faces_cache = array_cache(patch_cell_faces)
  faces_on_boundary_cache = array_cache(faces_on_boundary)
  for iC in 1:num_patch_cells
    cell_faces  = getindex!(patch_cells_faces_cache,patch_cell_faces,iC)
    on_boundary = getindex!(faces_on_boundary_cache,faces_on_boundary,iC)
    for (iF,face) in enumerate(cell_faces)
      if (!on_boundary[iF] && faces_mask[face])
        num_patch_faces += 1
      end
    end
  end

  patch_faces_data = zeros(Int64,num_patch_faces)
  patch_faces_ptrs = zeros(Int64,num_patch_cells+1)
  return Gridap.Arrays.Table(patch_faces_data,patch_faces_ptrs)
end

function _generate_patch_cell_faces!(patch_faces,patch_cell_faces,faces_on_boundary,faces_mask)
  num_patch_cells = length(patch_cell_faces)
  patch_faces_data, patch_faces_ptrs = patch_faces.data, patch_faces.ptrs

  pface = 1
  patch_faces_ptrs[1] = 1
  patch_cells_faces_cache = array_cache(patch_cell_faces)
  faces_on_boundary_cache = array_cache(faces_on_boundary)
  for iC in 1:num_patch_cells
    cell_faces  = getindex!(patch_cells_faces_cache,patch_cell_faces,iC)
    on_boundary = getindex!(faces_on_boundary_cache,faces_on_boundary,iC)
    patch_faces_ptrs[iC+1] = patch_faces_ptrs[iC]
    for (iF,face) in enumerate(cell_faces)
      if (!on_boundary[iF] && faces_mask[face])
        patch_faces_data[pface] = face
        patch_faces_ptrs[iC+1] += 1
        pface += 1
      end
    end
  end

  return patch_faces
end

"""
    get_patch_faces(PD::PatchDecomposition,Df::Integer,faces_mask::AbstractVector{Bool};reverse=false)

Returns a patch-wise Table containing the faces on each patch, i.e 

    patch_faces[patch] = [face1, face2, ...]

where face1, face2, ... are the faces on the patch such that 

  - they are NOT on the boundary of the patch
  - they are flagged `true` in `faces_mask`

If `reverse` is `true`, the faces are the ones ON the boundary of the patch.
"""
function get_patch_faces(
  PD::PatchDecomposition{Dr,Dc},Df::Integer,faces_mask::AbstractVector{Bool};reverse=false
) where {Dr,Dc}
  model    = PD.model
  topo     = get_grid_topology(model)

  c2e_map  = Gridap.Geometry.get_faces(topo,Dc,Df)
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  patch_cell_faces  = lazy_map(Reindex(c2e_map),patch_cells.data)
  faces_on_boundary = PD.patch_cells_faces_on_boundary[Df+1]

  patch_faces = _allocate_patch_faces(patch_cells,patch_cell_faces,faces_on_boundary,faces_mask,reverse)
  _generate_patch_faces!(patch_faces,patch_cells,patch_cell_faces,faces_on_boundary,faces_mask,reverse)

  return patch_faces
end

function _allocate_patch_faces(
  patch_cells,patch_cell_faces,faces_on_boundary,faces_mask,reverse
)
  num_patches = length(patch_cells)

  touched = Dict{Int,Bool}()
  pcell = 1
  num_patch_faces = 0
  patch_cells_cache       = array_cache(patch_cells)
  patch_cells_faces_cache = array_cache(patch_cell_faces)
  faces_on_boundary_cache = array_cache(faces_on_boundary)
  for patch in 1:num_patches
    current_patch_cells = getindex!(patch_cells_cache,patch_cells,patch)
    for _ in 1:length(current_patch_cells)
      cell_faces  = getindex!(patch_cells_faces_cache,patch_cell_faces,pcell)
      on_boundary = getindex!(faces_on_boundary_cache,faces_on_boundary,pcell)
      for (iF,face) in enumerate(cell_faces)
        A = xor(on_boundary[iF],reverse) # reverse the flag if needed
        if (!A && faces_mask[face] && !haskey(touched,face))
          num_patch_faces += 1
          touched[face] = true
        end
      end
      pcell += 1
    end
    empty!(touched)
  end

  patch_faces_data = zeros(Int64,num_patch_faces)
  patch_faces_ptrs = zeros(Int64,num_patches+1)
  return Gridap.Arrays.Table(patch_faces_data,patch_faces_ptrs)
end

function _generate_patch_faces!(
  patch_faces,patch_cells,patch_cell_faces,faces_on_boundary,faces_mask,reverse
)
  num_patches = length(patch_cells)
  patch_faces_data, patch_faces_ptrs = patch_faces.data, patch_faces.ptrs

  touched = Dict{Int,Bool}()
  pcell = 1
  pface = 1
  patch_faces_ptrs[1] = 1
  patch_cells_cache       = array_cache(patch_cells)
  patch_cells_faces_cache = array_cache(patch_cell_faces)
  faces_on_boundary_cache = array_cache(faces_on_boundary)
  for patch in 1:num_patches
    current_patch_cells = getindex!(patch_cells_cache,patch_cells,patch)
    patch_faces_ptrs[patch+1] = patch_faces_ptrs[patch]
    for _ in 1:length(current_patch_cells)
      cell_faces  = getindex!(patch_cells_faces_cache,patch_cell_faces,pcell)
      on_boundary = getindex!(faces_on_boundary_cache,faces_on_boundary,pcell)
      for (iF,face) in enumerate(cell_faces)
        A = xor(on_boundary[iF],reverse) # reverse the flag if needed
        if (!A && faces_mask[face] && !haskey(touched,face))
          patch_faces_data[pface] = face
          patch_faces_ptrs[patch+1] += 1
          touched[face] = true
          pface += 1
        end
      end
      pcell += 1
    end
    empty!(touched)
  end

  return patch_faces
end

"""
    get_pface_to_pcell(PD::PatchDecomposition{Dr,Dc},Df::Integer,patch_faces)

Returns two pface-wise Tables containing 

  1) the patch cells touched by each patch face and 
  2) the local cell index (within the face connectivity) of the cell touched by the patch face,
     which is needed when a pface touches different cells depending on the patch

i.e

    pface_to_pcell[pface] = [pcell1, pcell2, ...]
    pface_to_lcell[pface] = [lcell1, lcell2, ...]

where pcell1, pcell2, ... are the patch cells touched by the patch face `pface`.

This would be the Gridap equivalent to `get_faces(patch_topology,Df,Dc)`.
"""
function get_pface_to_pcell(PD::PatchDecomposition{Dr,Dc},Df::Integer,patch_faces) where {Dr,Dc}
  model    = PD.model
  topo     = get_grid_topology(model)

  faces_to_cells  = Gridap.Geometry.get_faces(topo,Df,Dc)
  pfaces_to_cells = lazy_map(Reindex(faces_to_cells),patch_faces.data)
  patch_cells     = Gridap.Arrays.Table(PD.patch_cells)
  patch_cells_overlapped = get_patch_cells_overlapped(PD)

  num_patches = length(patch_cells)
  num_pfaces  = length(pfaces_to_cells)

  patch_cells_cache = array_cache(patch_cells)
  patch_cells_overlapped_cache = array_cache(patch_cells_overlapped)
  pfaces_to_cells_cache = array_cache(pfaces_to_cells)

  # Maximum length of the data array
  k = sum(pface -> length(getindex!(pfaces_to_cells_cache,pfaces_to_cells,pface)),1:num_pfaces)

  # Collect patch cells touched by each pface
  # Remark: Each pface does NOT necessarily touch all it's neighboring cells, since 
  #         some of them might be outside the patch.
  ptrs = zeros(Int64,num_pfaces+1)
  data = zeros(Int64,k)
  pface_to_lcell = zeros(Int8,k)
  k = 1
  for patch in 1:num_patches
    cells = getindex!(patch_cells_cache,patch_cells,patch)
    cells_overlapped = getindex!(patch_cells_overlapped_cache,patch_cells_overlapped,patch)
    for pface in patch_faces.ptrs[patch]:patch_faces.ptrs[patch+1]-1
      pface_to_cells = getindex!(pfaces_to_cells_cache,pfaces_to_cells,pface)
      for (lcell,cell) in enumerate(pface_to_cells)
        lid_patch = findfirst(c->c==cell,cells)
        if !isnothing(lid_patch)
          ptrs[pface+1] += 1
          data[k] = cells_overlapped[lid_patch]
          pface_to_lcell[k] = Int8(lcell)
          k += 1
        end
      end
    end
  end
  Arrays.length_to_ptrs!(ptrs)
  data = resize!(data,k-1)
  pface_to_lcell = resize!(pface_to_lcell,k-1)

  pface_to_pcell = Gridap.Arrays.Table(data,ptrs)
  return pface_to_pcell, pface_to_lcell
end

"""
    generate_patch_closures(PD::PatchDecomposition{Dr,Dc})

Returns a patch-wise Table containing the closure of each patch.
"""
function generate_patch_closures(PD::PatchDecomposition{Dr,Dc}) where {Dr,Dc}
  topo = get_grid_topology(PD.model)
  nodes_to_cells = Geometry.get_faces(topo,0,Dc)
  cells_to_nodes = Geometry.get_faces(topo,Dc,0)

  patch_cells = get_patch_cells(PD)

  n_patches = length(patch_cells)
  ptrs = zeros(Int,n_patches+1)
  for patch in 1:n_patches
    cells = view(patch_cells,patch)
    closure = Set(cells)
    for cell in cells
      nodes = view(cells_to_nodes,cell)
      for node in nodes
        nbors = view(nodes_to_cells,node)
        push!(closure,nbors...)
      end
    end
    ptrs[patch+1] = length(closure)
  end
  Arrays.length_to_ptrs!(ptrs)

  data = zeros(Int,ptrs[end]-1)
  for patch in 1:n_patches
    cells = view(patch_cells,patch)
    
    # First we push the interior patch cells
    for cell in cells
      data[ptrs[patch]] = cell
      ptrs[patch] += 1
    end

    # Then we push the extra cells in the closure
    closure = Set(cells)
    for cell in cells
      nodes = view(cells_to_nodes,cell)
      for node in nodes
        nbors = view(nodes_to_cells,node)
        for nbor in nbors
          if nbor ∉ closure
            data[ptrs[patch]] = nbor
            push!(closure,nbor)
            ptrs[patch] += 1
          end
        end
      end
    end
  end
  Arrays.rewind_ptrs!(ptrs)

  return Table(data,ptrs)
end
