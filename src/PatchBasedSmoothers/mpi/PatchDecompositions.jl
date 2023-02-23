
struct DistributedPatchDecomposition{Dc,Dp,A,B} <: GridapType
  patch_decompositions::A
  model::B
end

GridapDistributed.local_views(a::DistributedPatchDecomposition) = a.patch_decompositions

function PatchDecomposition(model::GridapDistributed.AbstractDistributedDiscreteModel{Dc,Dp};
                            Dr=0,
                            patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude()) where {Dc,Dp}
  patch_decompositions = map_parts(local_views(model)) do lmodel
    PatchDecomposition(lmodel;
                       Dr=Dr,
                       patch_boundary_style=patch_boundary_style)
  end
  A = typeof(patch_decompositions)
  B = typeof(model)
  return DistributedPatchDecomposition{Dc,Dp,A,B}(patch_decompositions,model)
end

function PatchDecomposition(mh::ModelHierarchy;kwargs...)
  nlevs = num_levels(mh)
  decompositions = Vector{DistributedPatchDecomposition}(undef,nlevs-1)
  for lev in 1:nlevs-1
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model = get_model(mh,lev)
      decompositions[lev] = PatchDecomposition(model;kwargs...)
    end
  end
  return decompositions
end

function Gridap.Geometry.Triangulation(a::DistributedPatchDecomposition)
  trians = map_parts(a.patch_decompositions) do a
    Triangulation(a)
  end
  return GridapDistributed.DistributedTriangulation(trians,a.model)
end

function get_patch_root_dim(a::DistributedPatchDecomposition)
  patch_root_dim = -1
  map_parts(a.patch_decompositions) do patch_decomposition
    patch_root_dim = patch_decomposition.Dr
  end
  return patch_root_dim
end
