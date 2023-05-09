
struct PatchTriangulation{Dc,Dp,A,B,C} <: Gridap.Geometry.Triangulation{Dc,Dp}
  trian :: A
  PD    :: B
  patch_faces :: C

  function PatchTriangulation(trian::Triangulation{Dc,Dp},PD::PatchDecomposition,patch_faces) where {Dc,Dp}
    A = typeof(trian)
    B = typeof(PD)
    C = typeof(patch_faces)
    new{Dc,Dp,A,B,C}(trian,PD,patch_faces)
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
  trian = view(Triangulation(PD.model),patch_cells.data)
  return PatchTriangulation(trian,PD,patch_cells)
end

function Gridap.Geometry.BoundaryTriangulation(PD::PatchDecomposition{Dc}) where Dc
  Df       = Dc -1 
  model    = PD.model
  labeling = get_face_labeling(model)

  is_boundary = get_face_mask(labeling,["boundary"],Df)
  patch_edges = get_patch_cell_faces(PD,1,is_boundary)

  Γ    = BoundaryTriangulation(model)
  glue = get_glue(Γ,Val(Df))
  mface_to_tface = Gridap.Arrays.find_inverse_index_map(glue.tface_to_mface,num_faces(model,Df))
  patch_edges_data = lazy_map(Reindex(mface_to_tface),patch_edges.data)
  
  trian = view(Γ,patch_edges_data)
  return PatchTriangulation(trian,PD,patch_edges)
end

function Gridap.Geometry.SkeletonTriangulation(PD::PatchDecomposition{Dc}) where Dc
  Df       = Dc -1 
  model    = PD.model
  labeling = get_face_labeling(model)

  is_interior = get_face_mask(labeling,["interior"],Df)
  patch_edges = get_patch_cell_faces(PD,Df,is_interior)

  Λ    = SkeletonTriangulation(model)
  glue = get_glue(Λ,Val(Df))
  mface_to_tface   = Gridap.Arrays.find_inverse_index_map(glue.tface_to_mface,num_faces(model,Df))
  patch_edges_data = lazy_map(Reindex(mface_to_tface),patch_edges.data)
  
  trian = view(Λ,patch_edges_data)
  return PatchTriangulation(trian,PD,patch_edges)
end

# Integration 

function Gridap.Geometry.move_contributions(scell_to_val::AbstractArray,strian::PatchTriangulation)
  return move_contributions(scell_to_val,strian,strian.PD)
end

function Gridap.Geometry.move_contributions(
  scell_to_val::AbstractArray,
  strian::PatchTriangulation{Df},
  PD::PatchDecomposition{Dc}) where {Dc,Df}

  # If cell-wise triangulation, 
  if Df == Dc
    return scell_to_val, strian
  end

  # If not cell-wise, combine contributions in overlapped cells
  patch_faces = strian.patch_faces
  patch_faces_overlapped = Gridap.Arrays.Table(collect(1:length(patch_faces.data)),patch_faces.ptrs)
  _scell_to_val = lazy_map(Geometry.CombineContributionsMap(scell_to_val),patch_faces_overlapped)
  
  touched_cells = findall(map(i->patch_faces.ptrs[i] != patch_faces.ptrs[i+1],1:length(patch_faces)))
  touched_cell_to_val = lazy_map(Reindex(_scell_to_val),touched_cells)
  cell_trian = Triangulation(PD)
  touched_cell_trian = view(cell_trian,touched_cells)

  return touched_cell_to_val, touched_cell_trian
end
