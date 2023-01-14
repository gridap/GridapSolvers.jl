
struct DistributedPatchDecomposition{Dc,Dp,A,B} <: GridapType
  patch_decompositions::A
  model::B
end

function PatchDecomposition(model::GridapDistributed.DistributedDiscreteModel{Dc,Dp};
                            Dr=0,
                            patch_boundary_style::PatchBoundaryStyle=PatchBoundaryExclude()) where {Dc,Dp}
  patch_decompositions=map_parts(model.models) do lmodel
    PatchDecomposition(lmodel;
                       Dr=Dr,
                       patch_boundary_style=patch_boundary_style)
  end
  A=typeof(patch_decompositions)
  B=typeof(model)
  DistributedPatchDecomposition{Dc,Dp,A,B}(patch_decompositions,model)
end

function Gridap.Geometry.Triangulation(a::DistributedPatchDecomposition)
  trians=map_parts(a.patch_decompositions) do a
    Triangulation(a)
  end
  GridapDistributed.DistributedTriangulation(trians,a.model)
end

function get_patch_root_dim(a::DistributedPatchDecomposition)
  patch_root_dim=0
  map_parts(a.patch_decompositions) do patch_decomposition
    patch_root_dim=patch_decomposition.Dr
  end
  patch_root_dim
end
