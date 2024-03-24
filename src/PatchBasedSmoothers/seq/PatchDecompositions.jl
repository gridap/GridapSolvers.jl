abstract type PatchBoundaryStyle end
struct PatchBoundaryExclude  <: PatchBoundaryStyle end
struct PatchBoundaryInclude  <: PatchBoundaryStyle end

"""
PatchDecomposition{Dr,Dc,Dp} <: DiscreteModel{Dc,Dp}

  Dr :: Dimension of the patch root
  patch_cells                   :: [patch][local cell] -> cell
  patch_cells_overlapped        :: [patch][local cell] -> overlapped cell
  patch_cells_faces_on_boundary :: [d][overlapped cell][local face] -> face is on patch boundary
"""
struct PatchDecomposition{Dr,Dc,Dp} <: GridapType
  model                         :: DiscreteModel{Dc,Dp}
  patch_cells                   :: Gridap.Arrays.Table         # [patch][local cell] -> cell
  patch_cells_overlapped        :: Gridap.Arrays.Table         # [patch][local cell] -> overlapped cell
  patch_cells_faces_on_boundary :: Vector{Gridap.Arrays.Table} # [d][overlapped cell][local face] -> face is on patch boundary
end

num_patches(a::PatchDecomposition) = length(a.patch_cells)
Gridap.Geometry.num_cells(a::PatchDecomposition) = length(a.patch_cells.data)

function PatchDecomposition(
  model::DiscreteModel{Dc,Dp};
  Dr=0,
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude(),
  boundary_tag_names::AbstractArray{String}=["boundary"]
) where {Dc,Dp}
  Gridap.Helpers.@check 0 <= Dr <= Dc-1

  topology     = get_grid_topology(model)
  patch_cells  = Gridap.Geometry.get_faces(topology,Dr,Dc)
  patch_facets = Gridap.Geometry.get_faces(topology,Dr,Dc-1)
  patch_cells_overlapped = compute_patch_overlapped_cells(patch_cells)

  patch_cells_faces_on_boundary = compute_patch_cells_faces_on_boundary(
    model, patch_cells, patch_cells_overlapped,
    patch_facets, patch_boundary_style, boundary_tag_names
  )

  return PatchDecomposition{Dr,Dc,Dp}(
    model, patch_cells, patch_cells_overlapped, patch_cells_faces_on_boundary
  )
end

function compute_patch_overlapped_cells(patch_cells)
  num_overlapped_cells = length(patch_cells.data)
  data = Gridap.Arrays.IdentityVector(num_overlapped_cells)
  return Gridap.Arrays.Table(data,patch_cells.ptrs)
end

# patch_cell_faces_on_boundary :: 
#    [Df][overlapped cell][lface] -> Face is boundary of the patch
function compute_patch_cells_faces_on_boundary(
  model::DiscreteModel,
  patch_cells,
  patch_cells_overlapped,
  patch_facets,
  patch_boundary_style,
  boundary_tag_names
)
  patch_cell_faces_on_boundary = _allocate_patch_cells_faces_on_boundary(model,patch_cells)
  if !isa(patch_boundary_style,PatchBoundaryInclude)
    _compute_patch_cells_faces_on_boundary!(
      patch_cell_faces_on_boundary,
      model, patch_cells, patch_cells_overlapped,
      patch_facets, patch_boundary_style, boundary_tag_names
    )
  end
  return patch_cell_faces_on_boundary
end

function _allocate_patch_cells_faces_on_boundary(model::DiscreteModel{Dc},patch_cells) where {Dc}
  ctype_to_reffe = get_reffes(model)
  cell_to_ctype  = get_cell_type(model)
  
  patch_cells_faces_on_boundary = Vector{Gridap.Arrays.Table}(undef,Dc)
  for d = 0:Dc-1
    ctype_to_num_dfaces = map(reffe -> num_faces(reffe,d),ctype_to_reffe)
    patch_cells_faces_on_boundary[d+1] =
      _allocate_ocell_to_dface(Bool, patch_cells,cell_to_ctype, ctype_to_num_dfaces)
  end
  return patch_cells_faces_on_boundary
end

function _allocate_ocell_to_dface(::Type{T},patch_cells,cell_to_ctype,ctype_to_num_dfaces) where T<:Number
  num_overlapped_cells = length(patch_cells.data)
  ptrs = Vector{Int}(undef,num_overlapped_cells+1)

  ptrs[1] = 1
  for i = 1:num_overlapped_cells
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
  patch_cells_overlapped,
  patch_facets,
  patch_boundary_style,
  boundary_tag_names
)
  num_patches = length(patch_cells.ptrs)-1
  cache_patch_cells  = array_cache(patch_cells)
  cache_patch_facets = array_cache(patch_facets)
  for patch = 1:num_patches
    current_patch_cells  = getindex!(cache_patch_cells,patch_cells,patch)
    current_patch_facets = getindex!(cache_patch_facets,patch_facets,patch)
    _compute_patch_cells_faces_on_boundary!(
      patch_cells_faces_on_boundary,
      model,
      patch,
      current_patch_cells,
      patch_cells_overlapped,
      current_patch_facets,
      patch_boundary_style,
      boundary_tag_names
    )
  end
end

function _compute_patch_cells_faces_on_boundary!(
  patch_cells_faces_on_boundary,
  model::DiscreteModel{Dc},
  patch,
  patch_cells,
  patch_cells_overlapped,
  patch_facets,
  patch_boundary_style,
  boundary_tag_names
) where Dc
  face_labeling = get_face_labeling(model)
  topology = get_grid_topology(model)

  boundary_tags = findall(x -> (x ∈ boundary_tag_names), face_labeling.tag_to_name)
  Gridap.Helpers.@check !isempty(boundary_tags)
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
    overlapped_cell = patch_cells_overlapped.data[patch_cells_overlapped.ptrs[patch]+lcell-1]
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

                overlapped_cell2 = patch_cells_overlapped.data[patch_cells_overlapped.ptrs[patch]+lcell2-1]
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

# Patch cell faces: 
#   patch_faces[pcell] = [face1, face2, ...]
#   where face1, face2, ... are the faces of the overlapped cell `pcell` such that 
#      - they are NOT on the boundary of the patch
#      - they are flagged `true` in faces_mask
function get_patch_cell_faces(PD::PatchDecomposition,Df::Integer)
  model    = PD.model
  topo     = get_grid_topology(model)
  faces_mask = Fill(true,num_faces(topo,Df))
  return get_patch_cell_faces(PD,Df,faces_mask)
end

function get_patch_cell_faces(PD::PatchDecomposition{Dr,Dc},Df::Integer,faces_mask) where {Dr,Dc}
  model    = PD.model
  topo     = get_grid_topology(model)

  c2e_map  = Gridap.Geometry.get_faces(topo,Dc,Df)
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  patch_cell_faces  = map(Reindex(c2e_map),patch_cells.data)
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

# Patch faces: 
#   patch_faces[patch] = [face1, face2, ...]
#   where face1, face2, ... are the faces of the patch such that 
#      - they are NOT on the boundary of the patch
#      - they are flagged `true` in faces_mask
function get_patch_faces(PD::PatchDecomposition{Dr,Dc},Df::Integer,faces_mask) where {Dr,Dc}
  model    = PD.model
  topo     = get_grid_topology(model)

  c2e_map  = Gridap.Geometry.get_faces(topo,Dc,Df)
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  patch_cell_faces  = map(Reindex(c2e_map),patch_cells.data)
  faces_on_boundary = PD.patch_cells_faces_on_boundary[Df+1]

  patch_faces = _allocate_patch_faces(patch_cells,patch_cell_faces,faces_on_boundary,faces_mask)
  _generate_patch_faces!(patch_faces,patch_cells,patch_cell_faces,faces_on_boundary,faces_mask)

  return patch_faces
end

function _allocate_patch_faces(patch_cells,patch_cell_faces,faces_on_boundary,faces_mask)
  num_patches = length(patch_cells)

  touched = Dict{Int,Bool}()
  pcell = 1
  num_patch_faces = 0
  patch_cells_cache       = array_cache(patch_cells)
  patch_cells_faces_cache = array_cache(patch_cell_faces)
  faces_on_boundary_cache = array_cache(faces_on_boundary)
  for patch in 1:num_patches
    current_patch_cells = getindex!(patch_cells_cache,patch_cells,patch)
    for iC_local in 1:length(current_patch_cells)
      cell_faces  = getindex!(patch_cells_faces_cache,patch_cell_faces,pcell)
      on_boundary = getindex!(faces_on_boundary_cache,faces_on_boundary,pcell)
      for (iF,face) in enumerate(cell_faces)
        if (!on_boundary[iF] && faces_mask[face] && !haskey(touched,face))
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

function _generate_patch_faces!(patch_faces,patch_cells,patch_cell_faces,faces_on_boundary,faces_mask)
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
        if (!on_boundary[iF] && faces_mask[face] && !haskey(touched,face))
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

# Face connectivity for the patches
#    pfaces_to_pcells[pface] = [pcell1, pcell2, ...]
# This would be the Gridap equivalent to `get_faces(patch_topology,Df,Dc)`.
# The argument `patch_faces` allows to select only some pfaces (i.e boundary/skeleton/etc...).
function get_pfaces_to_pcells(PD::PatchDecomposition{Dr,Dc},Df::Integer,patch_faces) where {Dr,Dc}
  model    = PD.model
  topo     = get_grid_topology(model)

  faces_to_cells  = Gridap.Geometry.get_faces(topo,Df,Dc)
  pfaces_to_cells = lazy_map(Reindex(faces_to_cells),patch_faces.data)
  patch_cells     = Gridap.Arrays.Table(PD.patch_cells)
  patch_cells_overlapped = PD.patch_cells_overlapped

  num_patches = length(patch_cells)
  pf2pc_ptrs  = Gridap.Adaptivity.counts_to_ptrs(map(length,pfaces_to_cells))
  pf2pc_data  = zeros(Int64,pf2pc_ptrs[end]-1)

  patch_cells_cache = array_cache(patch_cells)
  patch_cells_overlapped_cache = array_cache(patch_cells_overlapped)
  pfaces_to_cells_cache = array_cache(pfaces_to_cells)
  for patch in 1:num_patches
    cells = getindex!(patch_cells_cache,patch_cells,patch)
    cells_overlapped = getindex!(patch_cells_overlapped_cache,patch_cells_overlapped,patch)
    for pface in patch_faces.ptrs[patch]:patch_faces.ptrs[patch+1]-1
      pface_to_cells = getindex!(pfaces_to_cells_cache,pfaces_to_cells,pface)
      for (lid,cell) in enumerate(pface_to_cells)
        lid_patch = findfirst(c->c==cell,cells)
        pf2pc_data[pf2pc_ptrs[pface]+lid-1] = cells_overlapped[lid_patch]
      end
    end
  end

  pfaces_to_pcells = Gridap.Arrays.Table(pf2pc_data,pf2pc_ptrs)
  return pfaces_to_pcells
end
