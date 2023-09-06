abstract type PatchBoundaryStyle end ;
struct PatchBoundaryExclude  <: PatchBoundaryStyle end ;
struct PatchBoundaryInclude  <: PatchBoundaryStyle end ;

# TODO: Make patch_cells a Table

# Question? Might a patch decomposition involve patches
#           with roots of different topological dimension?
#           This is not currently supported.
struct PatchDecomposition{Dc,Dp} <: GridapType
  model                         :: DiscreteModel{Dc,Dp}
  Dr                            :: Int # Topological dim of the face at the root of the patch
  patch_cells                   :: AbstractVector{<:AbstractVector} # Patch+local cell -> cell
  patch_cells_overlapped_mesh   :: Gridap.Arrays.Table # Patch+local cell -> overlapped cell
  patch_cells_faces_on_boundary :: Vector{Gridap.Arrays.Table} # Df + overlapped cell -> faces on
end

num_patches(a::PatchDecomposition) = length(a.patch_cells)
Gridap.Geometry.num_cells(a::PatchDecomposition) = a.patch_cells_overlapped_mesh.data[end]

function PatchDecomposition(
  model::DiscreteModel{Dc,Dp};
  Dr=0,
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude(),
  boundary_tag_names::AbstractArray{String}=["boundary"]) where {Dc,Dp}
  Gridap.Helpers.@check 0 <= Dr <= Dc-1

  grid               = get_grid(model)
  ctype_reffe        = get_reffes(grid)
  cell_type          = get_cell_type(grid)
  d_ctype_num_dfaces = [ map(reffe->num_faces(Gridap.Geometry.get_polytope(reffe),d),ctype_reffe) for d in 0:Dc]
  topology           = get_grid_topology(model)

  patch_cells  = Gridap.Geometry.get_faces(topology,Dr,Dc)
  patch_facets = Gridap.Geometry.get_faces(topology,Dr,Dc-1)
  patch_cells_overlapped_mesh = setup_patch_cells_overlapped_mesh(patch_cells)

  patch_cells_faces_on_boundary = allocate_patch_cells_faces_on_boundary(
                                          Dr,
                                          model,
                                          cell_type,
                                          d_ctype_num_dfaces,
                                          patch_cells,
                                          patch_cells_overlapped_mesh)

  generate_patch_boundary_faces!(patch_cells_faces_on_boundary,
                                 model,
                                 patch_cells,
                                 patch_cells_overlapped_mesh,
                                 patch_facets,
                                 patch_boundary_style,
                                 boundary_tag_names)

  return PatchDecomposition{Dc,Dp}(model, Dr,
                                   patch_cells,
                                   patch_cells_overlapped_mesh,
                                   patch_cells_faces_on_boundary)
end

function setup_patch_cells_overlapped_mesh(patch_cells)
  num_patches = length(patch_cells)
  cache = array_cache(patch_cells)
  ptrs  = Vector{Int}(undef,num_patches+1)
  ptrs[1] = 1
  for patch_id = 1:num_patches
    cells_around_patch = getindex!(cache,patch_cells,patch_id)
    ptrs[patch_id+1] = ptrs[patch_id] + length(cells_around_patch)
  end
  data = Gridap.Arrays.IdentityVector(ptrs[end]-1)
  return Gridap.Arrays.Table(data,ptrs)
end

function allocate_patch_cells_faces_on_boundary(Dr,
                                                model::DiscreteModel{Dc},
                                                cell_type,
                                                d_ctype_num_dfaces,
                                                patch_cells,
                                                patch_cells_overlapped_mesh) where {Dc}
  patch_cells_faces_on_boundary = Vector{Gridap.Arrays.Table}(undef,Dc)
  for d = 0:Dc-1
    patch_cells_faces_on_boundary[d+1] =
      allocate_cell_overlapped_mesh_lface(Bool, patch_cells, patch_cells_overlapped_mesh,
                                          cell_type, d_ctype_num_dfaces, d)
  end
  return patch_cells_faces_on_boundary
end

# Table 2
# position_of_cell_within_global_array -> sublist of entities associated to that
function allocate_cell_overlapped_mesh_lface(::Type{T},
                                             patch_cells,
                                             patch_cells_overlapped_mesh,
                                             cell_type,
                                             d_ctype_num_dfaces,
                                             dim) where T<:Number # dim=0,1,...,Dc-1
   n = length(patch_cells_overlapped_mesh.data) # number of cells in the overlapped mesh
   ptrs = Vector{Int}(undef,n+1)

   ptrs[1] = 1; n = 1
   for (patch,cells_patch) in enumerate(patch_cells)
     for cell in cells_patch
        ctype  = cell_type[cell]
        nfaces = d_ctype_num_dfaces[dim+1][ctype]
        # To get the cell in the non overlapped mesh
        ptrs[n+1] = ptrs[n] + nfaces
        n = n + 1
     end
   end
   data = zeros(T,ptrs[n]-1)
   return Gridap.Arrays.Table(data,ptrs)
end

function generate_patch_boundary_faces!(patch_cells_faces_on_boundary,
                                        model::DiscreteModel,
                                        patch_cells,
                                        patch_cells_overlapped_mesh,
                                        patch_facets,
                                        patch_boundary_style,
                                        boundary_tag_names)

    num_patches = length(patch_cells.ptrs)-1
    cache_patch_cells  = array_cache(patch_cells)
    cache_patch_facets = array_cache(patch_facets)
    for patch = 1:num_patches
      current_patch_cells  = getindex!(cache_patch_cells,patch_cells,patch)
      current_patch_facets = getindex!(cache_patch_facets,patch_facets,patch)
      generate_patch_boundary_faces!(patch_cells_faces_on_boundary,
                                     model,
                                     patch,
                                     current_patch_cells,
                                     patch_cells_overlapped_mesh,
                                     current_patch_facets,
                                     patch_boundary_style,
                                     boundary_tag_names)
    end
end

function generate_patch_boundary_faces!(patch_cells_faces_on_boundary,
                                        model::DiscreteModel{Dc},
                                        patch,
                                        patch_cells,
                                        patch_cells_overlapped_mesh,
                                        patch_facets,
                                        patch_boundary_style,
                                        boundary_tag_names) where Dc
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

  # Go over all cells in the current patch
  for (lpatch_cell,patch_cell) in enumerate(patch_cells)
    cell_facets = getindex!(cache_cell_to_facets,cell_to_facets,patch_cell)
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

      facet_at_global_boundary = (facet_entity ∈ boundary_entities)
      A = (facet_at_global_boundary) && (facet ∉ patch_facets)
      B = isa(patch_boundary_style,PatchBoundaryExclude) && has_nbor_outside_patch
      facet_at_patch_boundary = (A || B)

      if (facet_at_patch_boundary)
        overlapped_cell = patch_cells_overlapped_mesh[patch][lpatch_cell]
        position = patch_cells_faces_on_boundary[Df+1].ptrs[overlapped_cell]+lfacet-1
        patch_cells_faces_on_boundary[Df+1].data[position] = true

        # Go over the faces of the lower dimension on the boundary of
        # the facet. And then propagate true to all cells around, and
        # for each cell around, we need to identify which is the local
        # face identifier within that cell

        # Go over the faces on the boundary of the current facet
        for d = 0:Df-1
          d_faces_on_boundary_of_current_facet = Gridap.Geometry.get_faces(topology,Df,d)[facet]
          for f in d_faces_on_boundary_of_current_facet
            # # TO-DO: to use caches!!!
            # Locate the local position of f within the cell (lface)
            cells_d_faces = Gridap.Geometry.get_faces(topology,Dc,d)
            d_faces_cells = Gridap.Geometry.get_faces(topology,d,Dc)
            for cell_around_face in d_faces_cells[f]
              if (cell_around_face in patch_cells)
                cell_d_face   = cells_d_faces[cell_around_face]
                lface         = findfirst(x -> x==f, cell_d_face)
                lpatch_cell2  = findfirst(x -> x==cell_around_face, patch_cells)

                cell_overlapped_mesh = patch_cells_overlapped_mesh[patch][lpatch_cell2]
                position = patch_cells_faces_on_boundary[d+1].ptrs[cell_overlapped_mesh]+lface-1
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

function get_patch_cell_faces(PD::PatchDecomposition{Dc},Df::Integer,faces_mask) where Dc
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
function get_patch_faces(PD::PatchDecomposition{Dc},Df::Integer,faces_mask) where Dc
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
function get_pfaces_to_pcells(PD::PatchDecomposition{Dc},Df::Integer,patch_faces) where Dc
  model    = PD.model
  topo     = get_grid_topology(model)

  faces_to_cells  = Gridap.Geometry.get_faces(topo,Df,Dc)
  pfaces_to_cells = lazy_map(Reindex(faces_to_cells),patch_faces.data)
  patch_cells     = Gridap.Arrays.Table(PD.patch_cells)
  patch_cells_overlapped = PD.patch_cells_overlapped_mesh

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
