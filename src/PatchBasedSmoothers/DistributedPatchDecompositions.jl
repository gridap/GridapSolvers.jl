
struct DistributedPatchDecomposition{Dr,Dc,Dp,A,B} <: GridapType
  patch_decompositions::A
  model::B
  function DistributedPatchDecomposition(
    patch_decompositions::AbstractVector{<:PatchDecomposition{Dr,Dc,Dp}},
    model::GridapDistributed.DistributedDiscreteModel{Dc,Dp}
  ) where {Dr,Dc,Dp}
    A = typeof(patch_decompositions)
    B = typeof(model)
    new{Dr,Dc,Dp,A,B}(patch_decompositions,model)
  end
end

GridapDistributed.local_views(a::DistributedPatchDecomposition) = a.patch_decompositions

function PatchDecomposition(
  model::GridapDistributed.DistributedDiscreteModel{Dc,Dp};
  Dr=0,
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude()
) where {Dc,Dp}
  mark_interface_facets!(model)
  patch_decompositions = map(local_views(model)) do model
    PatchDecomposition(
      model;
      Dr=Dr,
      patch_boundary_style=patch_boundary_style,
      boundary_tag_names=["boundary","interface"]
    )
  end
  return DistributedPatchDecomposition(patch_decompositions,model)
end

function CoarsePatchDecomposition(
  model::GridapDistributed.DistributedAdaptedDiscreteModel{Dc,Dp};
  patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude()
) where {Dc,Dp}
  gids = get_cell_gids(Gridap.Adaptivity.get_parent(model))
  mark_interface_facets!(model)
  patch_decompositions = map(local_views(model),partition(gids)) do model, cids
    own_cells = own_to_local(cids)
    glue = Gridap.Adaptivity.get_adaptivity_glue(model)
    patch_cells = glue.o2n_faces_map[own_cells]
    PatchDecomposition(
      model,patch_cells,
      patch_boundary_style=patch_boundary_style,
      boundary_tag_names=["boundary","interface"]
    )
  end
  return DistributedPatchDecomposition(patch_decompositions,model)
end

function PatchDecomposition(mh::ModelHierarchy;kwargs...)
  nlevs = num_levels(mh)
  decomps = map(view(mh,1:nlevs-1)) do mhl
    model = get_model(mhl)
    PatchDecomposition(model;kwargs...)
  end
  return decomps
end

function Geometry.Triangulation(a::DistributedPatchDecomposition)
  trians = map(local_views(a)) do a
    Triangulation(a)
  end
  return GridapDistributed.DistributedTriangulation(trians,a.model)
end

function Geometry.BoundaryTriangulation(a::DistributedPatchDecomposition,args...;kwargs...)
  trians = map(local_views(a)) do a
    BoundaryTriangulation(a,args...;kwargs...)
  end
  return GridapDistributed.DistributedTriangulation(trians,a.model)
end

function Geometry.SkeletonTriangulation(a::DistributedPatchDecomposition,args...;kwargs...)
  trians = map(local_views(a)) do a
    SkeletonTriangulation(a,args...;kwargs...)
  end
  return GridapDistributed.DistributedTriangulation(trians,a.model)
end

get_patch_root_dim(::DistributedPatchDecomposition{Dr}) where Dr = Dr

function mark_interface_facets!(model::GridapDistributed.DistributedDiscreteModel{Dc,Dp}) where {Dc,Dp}
  face_labeling = get_face_labeling(model)
  topo = get_grid_topology(model)

  map(local_views(face_labeling),local_views(topo)) do face_labeling, topo
    tag_to_name = face_labeling.tag_to_name
    tag_to_entities = face_labeling.tag_to_entities
    d_to_dface_to_entity = face_labeling.d_to_dface_to_entity
  
    # Create new tag & entity
    interface_entity = maximum(map(x -> maximum(x;init=0),tag_to_entities)) + 1
    push!(tag_to_entities,[interface_entity])
    push!(tag_to_name,"interface")

    # Interface faces should also be interior
    interior_tag = findfirst(x->(x=="interior"),tag_to_name)
    push!(tag_to_entities[interior_tag],interface_entity)
  
    # Select interface entities
    boundary_tag = findfirst(x->(x=="boundary"),tag_to_name)
    boundary_entities = tag_to_entities[boundary_tag]
  
    f2c_map = Geometry.get_faces(topo,Dc-1,Dc)
    num_cells_around_facet = map(length,f2c_map)
    mx = maximum(num_cells_around_facet)
    for (f,nf) in enumerate(num_cells_around_facet)
      is_boundary = (d_to_dface_to_entity[Dc][f] ∈ boundary_entities)
      if !is_boundary && (nf != mx)
        d_to_dface_to_entity[Dc][f] = interface_entity
      end
    end
  end
end
