struct FESpaceHierarchyLevel{A,B,C}
  level               :: Int
  fe_space            :: A
  fe_space_red        :: B
  mh_level            :: C
end

"""
    const FESpaceHierarchy = HierarchicalArray{<:FESpaceHierarchyLevel}
  
A `FESpaceHierarchy` is a hierarchical array of `FESpaceHierarchyLevel` objects. It stores the 
adapted/redistributed fe spaces and the corresponding subcommunicators.

For convenience, implements some of the API of `FESpace`.
"""
const FESpaceHierarchy{T,A,B} = HierarchicalArray{<:T where T<:FESpaceHierarchyLevel,A,B}

FESpaces.get_fe_space(sh::FESpaceHierarchy,lev::Int) = get_fe_space(sh[lev])
FESpaces.get_fe_space(a::FESpaceHierarchyLevel{A,Nothing}) where {A} = a.fe_space
FESpaces.get_fe_space(a::FESpaceHierarchyLevel{A,B}) where {A,B} = a.fe_space_red

get_fe_space_before_redist(sh::FESpaceHierarchy,lev::Int) = get_fe_space_before_redist(sh[lev])
get_fe_space_before_redist(a::FESpaceHierarchyLevel) = a.fe_space

Adaptivity.get_model(sh::FESpaceHierarchy,level::Integer) = get_model(sh[level])
Adaptivity.get_model(a::FESpaceHierarchyLevel) = get_model(a.mh_level)

get_model_before_redist(a::FESpaceHierarchy,level::Integer) = get_model_before_redist(a[level])
get_model_before_redist(a::FESpaceHierarchyLevel) = get_model_before_redist(a.mh_level)

has_redistribution(sh::FESpaceHierarchy,level::Integer) = has_redistribution(sh[level])
has_redistribution(a::FESpaceHierarchyLevel) = has_redistribution(a.mh_level)

has_refinement(sh::FESpaceHierarchy,level::Integer) = has_refinement(sh[level])
has_refinement(a::FESpaceHierarchyLevel) = has_refinement(a.mh_level)

# Test/Trial FESpaces for ModelHierarchyLevels

function FESpaces.FESpace(mh::ModelHierarchyLevel,args...;kwargs...)
  if has_redistribution(mh)
    cparts, _ = get_old_and_new_parts(mh.red_glue,Val(false))
    Vh     = i_am_in(cparts) ? FESpace(get_model_before_redist(mh),args...;kwargs...) : nothing
    Vh_red = FESpace(get_model(mh),args...;kwargs...)
  else
    Vh = FESpace(get_model(mh),args...;kwargs...)
    Vh_red = nothing
  end
  return FESpaceHierarchyLevel(mh.level,Vh,Vh_red,mh)
end

function FESpaces.TrialFESpace(a::FESpaceHierarchyLevel,args...;kwargs...)
  Uh     = !isa(a.fe_space,Nothing) ? TrialFESpace(a.fe_space,args...;kwargs...) : nothing
  Uh_red = !isa(a.fe_space_red,Nothing) ? TrialFESpace(a.fe_space_red,args...;kwargs...) : nothing
  return FESpaceHierarchyLevel(a.level,Uh,Uh_red,a.mh_level)
end

# Test/Trial FESpaces for ModelHierarchies/FESpaceHierarchy

function FESpaces.FESpace(mh::ModelHierarchy,args...;kwargs...)
  map(mh) do mhl
    TestFESpace(mhl,args...;kwargs...)
  end
end

function FESpaces.FESpace(
  mh::ModelHierarchy,arg_vector::AbstractVector;kwargs...
)
  map(linear_indices(mh),mh) do l, mhl
    args = arg_vector[l]
    TestFESpace(mhl,args...;kwargs...)
  end
end

function FESpaces.TrialFESpace(sh::FESpaceHierarchy,u)
  map(sh) do shl
    TrialFESpace(shl,u)
  end
end

function FESpaces.TrialFESpace(sh::FESpaceHierarchy)
  map(TrialFESpace,sh)
end

# MultiField support

function Gridap.MultiField.MultiFieldFESpace(spaces::Vector{<:FESpaceHierarchyLevel};kwargs...)
  level  = spaces[1].level
  Uh     = all(map(s -> !isa(s.fe_space,Nothing),spaces)) ? MultiFieldFESpace(map(s -> s.fe_space, spaces); kwargs...) : nothing
  Uh_red = all(map(s -> !isa(s.fe_space_red,Nothing),spaces)) ? MultiFieldFESpace(map(s -> s.fe_space_red, spaces); kwargs...) : nothing
  return FESpaceHierarchyLevel(level,Uh,Uh_red,first(spaces).mh_level)
end

function Gridap.MultiField.MultiFieldFESpace(spaces::Vector{<:HierarchicalArray};kwargs...)
  @check all(s -> isa(s,FESpaceHierarchy),spaces)
  map(spaces...) do spaces_i...
    MultiFieldFESpace([spaces_i...];kwargs...)
  end
end

# Constant FESpaces

function FESpaces.ConstantFESpace(mh::ModelHierarchy)
  map(mh) do mhl
    ConstantFESpace(mhl)
  end
end

function FESpaces.ConstantFESpace(mh::MultilevelTools.ModelHierarchyLevel)
  reffe = ReferenceFE(lagrangian,Float64,0)
  if has_redistribution(mh)
    cparts, _ = get_old_and_new_parts(mh.red_glue,Val(false))
    Vh     = i_am_in(cparts) ? ConstantFESpace(get_model_before_redist(mh)) : nothing
    Vh_red = ConstantFESpace(get_model(mh))
  else
    Vh = ConstantFESpace(get_model(mh))
    Vh_red = nothing
  end
  return FESpaceHierarchyLevel(mh.level,Vh,Vh_red,mh)
end

# Computing system matrices

function compute_hierarchy_matrices(
  trials::FESpaceHierarchy,
  tests::FESpaceHierarchy,
  a::Function,
  l::Function,
  qdegree::Integer
)
  return compute_hierarchy_matrices(trials,tests,a,l,Fill(qdegree,num_levels(trials)))
end

function compute_hierarchy_matrices(
  trials::FESpaceHierarchy,
  tests::FESpaceHierarchy,
  a::Function,
  l::Function,
  qdegree::AbstractArray{<:Integer}
)
  mats, vecs = map(linear_indices(trials)) do lev
    model = get_model(trials,lev)
    U = get_fe_space(trials,lev)
    V = get_fe_space(tests,lev)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,qdegree[lev])
    ai(u,v) = a(u,v,dΩ)
    if lev == 1
      li(v) = l(v,dΩ)
      op    = AffineFEOperator(ai,li,U,V)
      return get_matrix(op), get_vector(op)
    else
      return assemble_matrix(ai,U,V), nothing
    end
  end |> tuple_of_arrays
  return mats, mats[1], vecs[1]
end
