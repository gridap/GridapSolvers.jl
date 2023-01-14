abstract type PatchBoundaryStyle end ;
struct PatchBoundaryExclude  <: PatchBoundaryStyle end ;
struct PatchBoundaryInclude  <: PatchBoundaryStyle end ;

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

num_patches(a::PatchDecomposition)= length(a.patch_cells_overlapped_mesh.ptrs)-1
Gridap.Geometry.num_cells(a::PatchDecomposition)  = a.patch_cells_overlapped_mesh.data[end]


function PatchDecomposition(
  model::DiscreteModel{Dc,Dp};
  Dr=0,
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude()) where {Dc,Dp}
  Gridap.Helpers.@check 0 <= Dr <= Dc-1

  grid               = get_grid(model)
  ctype_reffe        = get_reffes(grid)
  cell_type          = get_cell_type(grid)
  d_ctype_num_dfaces = [ map(reffe->num_faces(Gridap.Geometry.get_polytope(reffe),d),ctype_reffe) for d in 0:Dc]
  topology           = get_grid_topology(model)

  patch_cells=Gridap.Geometry.get_faces(topology,Dr,Dc)
  patch_facets=Gridap.Geometry.get_faces(topology,Dr,Dc-1)
  patch_cells_overlapped_mesh=
     setup_patch_cells_overlapped_mesh(patch_cells)

  patch_cells_faces_on_boundary = allocate_patch_cells_faces_on_boundary(
                                          Dr,
                                          model,
                                          cell_type,
                                          d_ctype_num_dfaces,
                                          patch_cells,
                                          patch_cells_overlapped_mesh)


  generate_patch_boundary_faces!(model,
                                 patch_cells_faces_on_boundary,
                                 patch_cells,
                                 patch_cells_overlapped_mesh,
                                 patch_facets,
                                 patch_boundary_style)

  PatchDecomposition{Dc,Dp}(model,
                            Dr,
                            patch_cells,
                            patch_cells_overlapped_mesh,
                            patch_cells_faces_on_boundary)
end

function Gridap.Geometry.Triangulation(a::PatchDecomposition)
   patch_cells=Gridap.Arrays.Table(a.patch_cells)
   view(Triangulation(a.model),patch_cells.data)
end

function setup_patch_cells_overlapped_mesh(patch_cells)
  num_patches=length(patch_cells)
  cache = array_cache(patch_cells)
  ptrs=Vector{Int}(undef,num_patches+1)
  ptrs[1]=1
  for patch_id=1:num_patches
    cells_around_patch=getindex!(cache,patch_cells,patch_id)
    ptrs[patch_id+1]=ptrs[patch_id]+length(cells_around_patch)
  end
  data=Gridap.Arrays.IdentityVector(ptrs[end]-1)
  Gridap.Arrays.Table(data,ptrs)
end


function allocate_patch_cells_faces_on_boundary(Dr,
                                                model::DiscreteModel{Dc},
                                                cell_type,
                                                d_ctype_num_dfaces,
                                                patch_cells,
                                                patch_cells_overlapped_mesh) where {Dc}
  patch_cells_faces_on_boundary = Vector{Gridap.Arrays.Table}(undef,Dc)
  for d=0:Dc-1
    patch_cells_faces_on_boundary[d+1]=
      allocate_cell_overlapped_mesh_lface(Bool,
                                          patch_cells,
                                          patch_cells_overlapped_mesh,
                                          cell_type,
                                          d_ctype_num_dfaces,
                                          d)
  end
  patch_cells_faces_on_boundary
end

# Table 2
# position_of_cell_within_global_array -> sublist of entities associated to that
function allocate_cell_overlapped_mesh_lface(::Type{T},
                                             patch_cells,
                                             patch_cells_overlapped_mesh,
                                             cell_type,
                                             d_ctype_num_dfaces,
                                             dim) where T<:Number # dim=0,1,...,Dc-1
   n=length(patch_cells_overlapped_mesh.data) # number of cells in the overlapped mesh
   ptrs=Vector{Int}(undef,n+1)
   ptrs[1]=1
   n=1
   for patch=1:length(patch_cells)
     cells_patch=patch_cells[patch]
     for cell in cells_patch
       ctype  = cell_type[cell]
       nfaces = d_ctype_num_dfaces[dim+1][ctype]
       # To get the cell in the non overlapped mesh
       ptrs[n+1]=ptrs[n]+nfaces
       n=n+1
     end
   end
   data=zeros(T,ptrs[n]-1)
   Gridap.Arrays.Table(data,ptrs)
end

function generate_patch_boundary_faces!(model,
                                        patch_cells_faces_on_boundary,
                                        patch_cells,
                                        patch_cells_overlapped_mesh,
                                        patch_facets,
                                        patch_boundary_style)
    Dc=num_cell_dims(model)
    topology=get_grid_topology(model)
    labeling=get_face_labeling(model)
    num_patches=length(patch_cells.ptrs)-1
    cache_patch_cells=array_cache(patch_cells)
    cache_patch_facets=array_cache(patch_facets)
    for patch=1:num_patches
      current_patch_cells=getindex!(cache_patch_cells,patch_cells,patch)
      current_patch_facets=getindex!(cache_patch_facets,patch_facets,patch)
      generate_patch_boundary_faces!(patch_cells_faces_on_boundary,
                                     Dc,
                                     topology,
                                     labeling,
                                     patch,
                                     current_patch_cells,
                                     patch_cells_overlapped_mesh,
                                     current_patch_facets,
                                     patch_boundary_style)
    end
end

function generate_patch_boundary_faces!(patch_cells_faces_on_boundary,
                                        Dc,
                                        topology,
                                        face_labeling,
                                        patch,
                                        patch_cells,
                                        patch_cells_overlapped_mesh,
                                        patch_facets,
                                        patch_boundary_style)

  boundary_tag=findfirst(x->(x=="boundary"),face_labeling.tag_to_name)
  Gridap.Helpers.@check boundary_tag != nothing
  boundary_entities=face_labeling.tag_to_entities[boundary_tag]

  # Cells facets
  Df=Dc-1
  cells_facets=Gridap.Geometry.get_faces(topology,Dc,Df)
  cache_cells_facets=array_cache(cells_facets)

  # Cells around facets
  cells_around_facets=Gridap.Geometry.get_faces(topology,Df,Dc)
  cache_cells_around_facets=array_cache(cells_around_facets)

  # Go over all cells in the current patch
  for (lpatch_cell,patch_cell) in enumerate(patch_cells)
    cell_facets=getindex!(cache_cells_facets,cells_facets,patch_cell)
    # Go over the facets (i.e., faces of dim D-1) in the current cell
    for (lfacet,facet) in enumerate(cell_facets)
      facet_entity=face_labeling.d_to_dface_to_entity[Df+1][facet]

      cells_around_facet=getindex!(cache_cells_around_facets,
                                   cells_around_facets,
                                   facet)

      # Go over the cells around facet
      cell_not_in_patch_found=false
      for cell_around_facet in cells_around_facet
        if !(cell_around_facet in patch_cells)
          cell_not_in_patch_found=true
          break
        end
      end

      facet_at_global_boundary = facet_entity in boundary_entities
      if (facet_at_global_boundary)
        if (facet in patch_facets)
          facet_at_patch_boundary = false
        else
          facet_at_patch_boundary = true
        end
      elseif (patch_boundary_style isa PatchBoundaryInclude)
        facet_at_patch_boundary = false
      elseif ((patch_boundary_style  isa PatchBoundaryExclude) && cell_not_in_patch_found)
        facet_at_patch_boundary = true
      else
        facet_at_patch_boundary = false
      end

      # if (facet_at_neumann_boundary)
      #     println("XXX")
      #     println(facet)
      #     println(length(cells_around_facet))
      #     @assert length(cells_around_facet)==1
      #     println(cell_not_in_patch_found)
      #     @assert !cell_not_in_patch_found
      #     println("YYY")
      #     @assert !facet_at_boundary
      # end

      if (facet_at_patch_boundary)
        cell_overlapped_mesh = patch_cells_overlapped_mesh[patch][lpatch_cell]
        position=patch_cells_faces_on_boundary[Df+1].ptrs[cell_overlapped_mesh]+lfacet-1
        patch_cells_faces_on_boundary[Df+1].data[position]=true

        # Go over the faces of the lower dimension on the boundary of
        # the facet. And then propagate true to all cells around, and
        # for each cell around, we need to identify which is the local
        # face identifier within that cell

        # Go over the faces on the boundary of the current facet
        for d=0:Df-1
          d_faces_on_boundary_of_current_facet=Gridap.Geometry.get_faces(topology,Df,d)[facet]
          for f in d_faces_on_boundary_of_current_facet
            # # TO-DO: to use caches!!!
            # Locate the local position of f within the cell (lface)
            cells_d_faces = Gridap.Geometry.get_faces(topology,Dc,d)
            d_faces_cells = Gridap.Geometry.get_faces(topology,d,Dc)
            for cell_around_face in d_faces_cells[f]
              if (cell_around_face in patch_cells)
                cell_d_face   = cells_d_faces[cell_around_face]
                lface         = findfirst((x->x==f),cell_d_face)
                lpatch_cell2   = findfirst((x->x==cell_around_face),patch_cells)
                cell_overlapped_mesh =
                    patch_cells_overlapped_mesh[patch][lpatch_cell2]
                position=patch_cells_faces_on_boundary[d+1].ptrs[cell_overlapped_mesh]+lface-1
                patch_cells_faces_on_boundary[d+1].data[position]=true
              end
            end
          end
        end
      end
    end
  end
end
