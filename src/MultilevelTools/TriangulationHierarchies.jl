
struct TriangulationHierarchyLevel{A,B,C}
  level     :: Int
  trian     :: A
  trian_red :: B
  mh_level  :: C
end

const TriangulationHierarchy{T,A,B} = HierarchicalArray{<:T where T<:TriangulationHierarchyLevel,A,B}

Geometry.get_triangulation(sh::TriangulationHierarchy,lev::Int) = get_triangulation(sh[lev])
Geometry.get_triangulation(a::TriangulationHierarchyLevel{A,Nothing}) where {A} = a.trian
Geometry.get_triangulation(a::TriangulationHierarchyLevel{A,B}) where {A,B} = a.trian_red

get_triangulation_before_redist(sh::TriangulationHierarchy,lev::Int) = get_triangulation_before_redist(sh[lev])
get_triangulation_before_redist(a::TriangulationHierarchyLevel) = a.trian

has_redistribution(sh::TriangulationHierarchy,level::Integer) = has_redistribution(sh[level])
has_redistribution(a::TriangulationHierarchyLevel) = has_redistribution(a.mh_level)

has_refinement(sh::TriangulationHierarchy,level::Integer) = has_refinement(sh[level])
has_refinement(a::TriangulationHierarchyLevel) = has_refinement(a.mh_level)

function Geometry.Triangulation(::Type{ReferenceFE{d}},mh::ModelHierarchyLevel,args...;kwargs...) where {d}
  if has_redistribution(mh)
    cparts, _ = get_old_and_new_parts(mh.red_glue,Val(false))
    trian     = i_am_in(cparts) ? Triangulation(ReferenceFE{d},get_model_before_redist(mh),args...;kwargs...) : nothing
    trian_red = Triangulation(ReferenceFE{d},get_model(mh),args...;kwargs...)
  else
    trian = Triangulation(ReferenceFE{d},get_model(mh),args...;kwargs...)
    trian_red = nothing
  end
  return TriangulationHierarchyLevel(mh.level,trian,trian_red,mh)
end

function Geometry.Triangulation(::Type{ReferenceFE{d}},mh::ModelHierarchy,args...;kwargs...) where {d}
  map(mh) do mhl
    Triangulation(ReferenceFE{d},mhl,args...;kwargs...)
  end
end

function Geometry.Triangulation(mh::ModelHierarchy,args...;kwargs...)
  d = num_cell_dims(get_model(mh,1))
  Geometry.Triangulation(ReferenceFE{d},mh,args...;kwargs...)
end
