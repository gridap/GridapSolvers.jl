"""
    struct PatchTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}
      ...
    end

Wrapper around a Triangulation, for patch-based assembly.
"""
struct PatchTriangulation{Dc,Dp,A,B,C,D,E} <: Gridap.Geometry.Triangulation{Dc,Dp}
  trian            :: A
  PD               :: B
  patch_faces      :: C
  pfaces_to_pcells :: D
  mface_to_tface   :: E

  function PatchTriangulation(trian::Triangulation{Dc,Dp},
                              PD::PatchDecomposition,
                              patch_faces,pfaces_to_pcells,mface_to_tface) where {Dc,Dp}
    A = typeof(trian)
    B = typeof(PD)
    C = typeof(patch_faces)
    D = typeof(pfaces_to_pcells)
    E = typeof(mface_to_tface)
    new{Dc,Dp,A,B,C,D,E}(trian,PD,patch_faces,pfaces_to_pcells,mface_to_tface)
  end
end

# Triangulation API

function Geometry.get_background_model(t::PatchTriangulation)
  get_background_model(t.trian)
end

function Geometry.get_grid(t::PatchTriangulation)
  get_grid(t.trian)
end

function Geometry.get_glue(t::PatchTriangulation,::Val{d}) where d
  get_glue(t.trian,Val(d))
end

function Geometry.get_facet_normal(trian::PatchTriangulation)
  get_facet_normal(trian.trian)
end

# Constructors 

function Gridap.Geometry.Triangulation(PD::PatchDecomposition)
  patch_cells = Gridap.Arrays.Table(PD.patch_cells)
  trian = Triangulation(PD.model)
  return PatchTriangulation(trian,PD,patch_cells,nothing,nothing)
end

function Gridap.Geometry.BoundaryTriangulation(PD::PatchDecomposition{Dr,Dc};tags="boundary") where {Dr,Dc}
  Df       = Dc-1 
  model    = PD.model
  labeling = get_face_labeling(model)

  is_boundary = get_face_mask(labeling,tags,Df)
  patch_faces = get_patch_faces(PD,Df,is_boundary)
  pfaces_to_pcells = get_pfaces_to_pcells(PD,Df,patch_faces)

  trian = BoundaryTriangulation(model;tags)
  glue  = get_glue(trian,Val(Df))
  mface_to_tface = Gridap.Arrays.find_inverse_index_map(glue.tface_to_mface,num_faces(model,Df))
  
  return PatchTriangulation(trian,PD,patch_faces,pfaces_to_pcells,mface_to_tface)
end

function Gridap.Geometry.SkeletonTriangulation(PD::PatchDecomposition{Dr,Dc}) where {Dr,Dc}
  Df       = Dc-1 
  model    = PD.model
  labeling = get_face_labeling(model)

  is_interior = get_face_mask(labeling,["interior"],Df)
  patch_faces = get_patch_faces(PD,Df,is_interior)
  pfaces_to_pcells = get_pfaces_to_pcells(PD,Df,patch_faces)

  trian = SkeletonTriangulation(model)
  glue  = get_glue(trian,Val(Df))
  mface_to_tface = Gridap.Arrays.find_inverse_index_map(glue.tface_to_mface,num_faces(model,Df))
  
  return PatchTriangulation(trian,PD,patch_faces,pfaces_to_pcells,mface_to_tface)
end

# Move contributions

function Gridap.Geometry.move_contributions(scell_to_val::AbstractArray,strian::PatchTriangulation)
  return move_contributions(strian.trian,scell_to_val,strian)
end

function Gridap.Geometry.move_contributions(t::Gridap.Adaptivity.AdaptedTriangulation,
                                            scell_to_val::AbstractArray,
                                            strian::PatchTriangulation)
  return move_contributions(t.trian,scell_to_val,strian)
end

function Gridap.Geometry.move_contributions(::Triangulation,
                                            scell_to_val::AbstractArray,
                                            strian::PatchTriangulation)
  patch_cells = strian.patch_faces
  return lazy_map(Reindex(scell_to_val),patch_cells.data), strian
end

function Gridap.Geometry.move_contributions(::Union{<:BoundaryTriangulation,<:SkeletonTriangulation},
                                            scell_to_val::AbstractArray,
                                            strian::PatchTriangulation)
  patch_faces      = strian.patch_faces
  mface_to_tface   = strian.mface_to_tface
  patch_faces_data = lazy_map(Reindex(mface_to_tface),patch_faces.data)
  return lazy_map(Reindex(scell_to_val),patch_faces_data), strian
end
