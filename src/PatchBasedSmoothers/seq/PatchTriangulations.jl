"""
    struct PatchTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}
      ...
    end

Wrapper around a Triangulation, for patch-based assembly.
"""
struct PatchTriangulation{Dc,Dp,A,B,C,D} <: Triangulation{Dc,Dp}
  trian          :: A
  PD             :: B
  patch_faces    :: C
  pface_to_pcell :: D

  function PatchTriangulation(
    trian::Triangulation{Dc,Dp},
    PD::PatchDecomposition,
    patch_faces,
    pface_to_pcell
  ) where {Dc,Dp}
    A = typeof(trian)
    B = typeof(PD)
    C = typeof(patch_faces)
    D = typeof(pface_to_pcell)
    new{Dc,Dp,A,B,C,D}(trian,PD,patch_faces,pface_to_pcell)
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

# For now, I am disabling changes from PatchTriangulations to other Triangulations.
# Reason: When the tface_to_mface map is not injective (i.e when we have overlapping), 
#         the glue is not well defined. Gridap will throe an error when trying to create 
#         the inverse map mface_to_tface.
# I believe this could technically be relaxed in the future, but for now I don't see a 
# scenario where we would need this.
function Geometry.is_change_possible(strian::PatchTriangulation,ttrian::Triangulation)
  return strian === ttrian
end

# Constructors 

function Geometry.Triangulation(PD::PatchDecomposition)
  patch_cells = PD.patch_cells
  trian = Triangulation(PD.model)
  return PatchTriangulation(trian,PD,patch_cells,nothing)
end

# By default, we return faces which are NOT patch-boundary faces. To get the patch-boundary faces,
# set `reverse` to true (see docs for `get_patch_faces`).
function Geometry.BoundaryTriangulation(
  PD::PatchDecomposition{Dr,Dc};tags="boundary",reverse=false,
) where {Dr,Dc}
  Df       = Dc-1 
  model    = PD.model
  labeling = get_face_labeling(model)

  is_boundary = get_face_mask(labeling,tags,Df)
  patch_faces = get_patch_faces(PD,Df,is_boundary;reverse)
  pface_to_pcell, pface_to_lcell = get_pface_to_pcell(PD,Df,patch_faces)

  trian = OverlappingBoundaryTriangulation(model,patch_faces.data,pface_to_lcell)

  return PatchTriangulation(trian,PD,patch_faces,pface_to_pcell)
end

function Geometry.SkeletonTriangulation(PD::PatchDecomposition{Dr,Dc}) where {Dr,Dc}
  Df       = Dc-1 
  model    = PD.model
  labeling = get_face_labeling(model)

  is_interior = get_face_mask(labeling,["interior"],Df)
  patch_faces = get_patch_faces(PD,Df,is_interior)
  pface_to_pcell, _ = get_pface_to_pcell(PD,Df,patch_faces)

  nfaces = length(patch_faces.data)
  plus  = OverlappingBoundaryTriangulation(model,patch_faces.data,fill(Int8(1),nfaces))
  minus = OverlappingBoundaryTriangulation(model,patch_faces.data,fill(Int8(2),nfaces))
  trian = SkeletonTriangulation(plus,minus)

  return PatchTriangulation(trian,PD,patch_faces,pface_to_pcell)
end

# Move contributions

@inline function Geometry.move_contributions(scell_to_val::AbstractArray,strian::PatchTriangulation)
  return move_contributions(strian.trian,scell_to_val,strian)
end

function Geometry.move_contributions(
  t::Gridap.Adaptivity.AdaptedTriangulation,
  scell_to_val::AbstractArray,
  strian::PatchTriangulation
)
  return move_contributions(t.trian,scell_to_val,strian)
end

function Geometry.move_contributions(
  ::BodyFittedTriangulation,
  scell_to_val::AbstractArray,
  strian::PatchTriangulation
)
  patch_cells = strian.patch_faces
  return lazy_map(Reindex(scell_to_val),patch_cells.data), strian
end

function Geometry.move_contributions(
  ::Union{<:BoundaryTriangulation,<:SkeletonTriangulation},
  scell_to_val::AbstractArray,
  strian::PatchTriangulation
)
  display(ndims(eltype(scell_to_val)))
  return scell_to_val, strian
end


# Overlapping BoundaryTriangulation
#
# This is the situation: I do not see any show-stopper for us to have an overlapping
# BoundaryTriangulation. Within the FaceToCellGlue, the array `bgface_to_lcell` is never 
# used for anything else than the constructor. 
# So in my mind nothing stops us from creating the glue from a `face_to_lcell` array instead.
# 
# The following code does just that, and returns a regular BoundaryTriangulation. It is 
# mostly copied from Gridap/Geometry/BoundaryTriangulations.jl

function OverlappingBoundaryTriangulation(
  model::DiscreteModel,
  face_to_bgface::AbstractVector{<:Integer},
  face_to_lcell::AbstractVector{<:Integer}
)
  D = num_cell_dims(model)
  topo = get_grid_topology(model)
  bgface_grid = Grid(ReferenceFE{D-1},model)

  face_grid = view(bgface_grid,face_to_bgface)
  cell_grid = get_grid(model)
  glue  = OverlappingFaceToCellGlue(topo,cell_grid,face_grid,face_to_bgface,face_to_lcell)
  trian = BodyFittedTriangulation(model,face_grid,face_to_bgface)

  return BoundaryTriangulation(trian,glue)
end

function OverlappingFaceToCellGlue(
  topo::GridTopology,
  cell_grid::Grid,
  face_grid::Grid,
  face_to_bgface::AbstractVector,
  face_to_lcell::AbstractVector
)
  Dc = num_cell_dims(cell_grid)
  Df = num_cell_dims(face_grid)
  bgface_to_cell = get_faces(topo,Df,Dc)
  bgcell_to_bgface = get_faces(topo,Dc,Df)
  cell_to_lface_to_pindex = Table(get_cell_permutations(topo,Df))

  face_to_cell = lazy_map(Reindex(bgface_to_cell), face_to_bgface)
  face_to_cell = collect(Int32,lazy_map(getindex,face_to_cell,face_to_lcell))
  face_to_lface = overlapped_find_local_index(face_to_bgface,face_to_cell,bgcell_to_bgface)

  f = (p)->fill(Int8(UNSET),num_faces(p,Df))
  ctype_to_lface_to_ftype = map( f, get_reffes(cell_grid) )
  face_to_ftype = get_cell_type(face_grid)
  cell_to_ctype = get_cell_type(cell_grid)

  Geometry._fill_ctype_to_lface_to_ftype!(
    ctype_to_lface_to_ftype,
    face_to_cell,
    face_to_lface,
    face_to_ftype,
    cell_to_ctype)

  Geometry.FaceToCellGlue(
    face_to_bgface,
    face_to_lcell,
    face_to_cell,
    face_to_lface,
    face_to_lcell,
    face_to_ftype,
    cell_to_ctype,
    cell_to_lface_to_pindex,
    ctype_to_lface_to_ftype
  )
end

function overlapped_find_local_index(
  c_to_a :: Vector{<:Integer},
  c_to_b :: Vector{<:Integer},
  b_to_lc_to_a :: Table
)
  c_to_lc = fill(Int8(-1),length(c_to_a))
  for (c,a) in enumerate(c_to_a)
    b = c_to_b[c]
    pini = b_to_lc_to_a.ptrs[b]
    pend = b_to_lc_to_a.ptrs[b+1]-1
    for (lc,p) in enumerate(pini:pend)
      if a == b_to_lc_to_a.data[p]
        c_to_lc[c] = Int8(lc)
        break
      end
    end
  end
  return c_to_lc
end
