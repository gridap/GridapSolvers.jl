struct FESpaceHierarchyLevel{A,B,C}
  level           :: Int
  fe_space        :: A
  fe_space_red    :: B
  cell_conformity :: C
end

struct FESpaceHierarchy
  mh     :: ModelHierarchy
  levels :: Vector{FESpaceHierarchyLevel}
end


function Base.getindex(fh::FESpaceHierarchy,level::Integer)
  fh.levels[level]
end

function Base.length(fh::FESpaceHierarchy)
  length(fh.levels)
end

function num_levels(fh::FESpaceHierarchy)
  length(fh)
end

function get_fe_space(a::FESpaceHierarchyLevel{A,Nothing}) where {A}
  a.fe_space
end

function get_fe_space(a::FESpaceHierarchyLevel{A,B}) where {A,B}
  a.fe_space_red
end

function get_fe_space(fh::FESpaceHierarchy,lev::Int)
  get_fe_space(fh[lev])
end

function get_fe_space_before_redist(a::FESpaceHierarchyLevel)
  a.fe_space
end

function get_fe_space_before_redist(fh::FESpaceHierarchy,lev::Int)
  get_fe_space_before_redist(fh[lev])
end

# Test/Trial FESpaces for ModelHierarchyLevels

function _cell_conformity(model::DiscreteModel,
                          reffe::Tuple{<:Gridap.FESpaces.ReferenceFEName,Any,Any}; 
                          conformity=nothing, kwargs...)
  basis, reffe_args, reffe_kwargs = reffe
  cell_reffe = ReferenceFE(model,basis,reffe_args...;reffe_kwargs...)
  conformity = Conformity(Gridap.Arrays.testitem(cell_reffe),conformity)
  return CellConformity(cell_reffe,conformity)
end

function _cell_conformity(model::GridapDistributed.DistributedDiscreteModel,args...;kwargs...)
  cell_conformities = map(local_views(model)) do model
    _cell_conformity(model,args...;kwargs...)
  end
  return cell_conformities
end

function Gridap.FESpaces.FESpace(
      mh::ModelHierarchyLevel{A,B,C,Nothing},args...;kwargs...) where {A,B,C}
  Vh = FESpace(get_model(mh),args...;kwargs...)
  cell_conformity = _cell_conformity(get_model(mh),args...;kwargs...)
  return FESpaceHierarchyLevel(mh.level,Vh,nothing,cell_conformity)
end

function Gridap.FESpaces.FESpace(mh::ModelHierarchyLevel{A,B,C,D},args...;kwargs...) where {A,B,C,D}
  cparts, _ = get_old_and_new_parts(mh.red_glue,Val(false))
  Vh     = i_am_in(cparts) ? FESpace(get_model_before_redist(mh),args...;kwargs...) : nothing
  Vh_red = FESpace(get_model(mh),args...;kwargs...)
  cell_conformity = _cell_conformity(get_model(mh),args...;kwargs...)
  return FESpaceHierarchyLevel(mh.level,Vh,Vh_red,cell_conformity)
end

function Gridap.FESpaces.TrialFESpace(a::FESpaceHierarchyLevel,args...;kwargs...)
  Uh     = !isa(a.fe_space,Nothing) ? TrialFESpace(a.fe_space,args...;kwargs...) : nothing
  Uh_red = !isa(a.fe_space_red,Nothing) ? TrialFESpace(a.fe_space_red,args...;kwargs...) : nothing
  return FESpaceHierarchyLevel(a.level,Uh,Uh_red,a.cell_conformity)
end

# Test/Trial FESpaces for ModelHierarchies/FESpaceHierarchy

function Gridap.FESpaces.FESpace(mh::ModelHierarchy,args...;kwargs...)
  test_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(mh))
  for i = 1:num_levels(mh)
    parts = get_level_parts(mh,i)
    if i_am_in(parts)
      Vh = TestFESpace(get_level(mh,i),args...;kwargs...)
      test_spaces[i] = Vh
    end
  end
  FESpaceHierarchy(mh,test_spaces)
end

function Gridap.FESpaces.FESpace(
                      mh::ModelHierarchy,
                      arg_vector::AbstractVector{<:Union{ReferenceFE,Tuple{<:Gridap.ReferenceFEs.ReferenceFEName,Any,Any}}};
                      kwargs...)
  @check length(arg_vector) == num_levels(mh)
  test_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(mh))
  for i = 1:num_levels(mh)
    parts = get_level_parts(mh,i)
    if i_am_in(parts)
      args = arg_vector[i]
      Vh   = TestFESpace(get_level(mh,i),args;kwargs...)
      test_spaces[i] = Vh
    end
  end
  FESpaceHierarchy(mh,test_spaces)
end

function Gridap.FESpaces.TrialFESpace(a::FESpaceHierarchy,u)
  trial_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(a.mh))
  for i = 1:num_levels(a.mh)
    parts = get_level_parts(a.mh,i)
    if i_am_in(parts)
      Uh = TrialFESpace(a[i],u)
      trial_spaces[i] = Uh
    end
  end
  FESpaceHierarchy(a.mh,trial_spaces)
end

function Gridap.FESpaces.TrialFESpace(a::FESpaceHierarchy)
  trial_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(a.mh))
  for i = 1:num_levels(a.mh)
    parts = get_level_parts(a.mh,i)
    if i_am_in(parts)
      Uh = TrialFESpace(a[i])
      trial_spaces[i] = Uh
    end
  end
  FESpaceHierarchy(a.mh,trial_spaces)
end

# MultiField support

function Gridap.MultiField.MultiFieldFESpace(spaces::Vector{<:FESpaceHierarchyLevel};kwargs...)
  level  = spaces[1].level
  Uh     = all(map(s -> !isa(s.fe_space,Nothing),spaces)) ? MultiFieldFESpace(map(s -> s.fe_space, spaces); kwargs...) : nothing
  Uh_red = all(map(s -> !isa(s.fe_space_red,Nothing),spaces)) ? MultiFieldFESpace(map(s -> s.fe_space_red, spaces); kwargs...) : nothing
  cell_conformity = map(s -> s.cell_conformity, spaces)
  return FESpaceHierarchyLevel(level,Uh,Uh_red,cell_conformity)
end

function Gridap.MultiField.MultiFieldFESpace(spaces::Vector{<:FESpaceHierarchy};kwargs...)
  mh = spaces[1].mh
  levels = Vector{FESpaceHierarchyLevel}(undef,num_levels(mh))
  for i = 1:num_levels(mh)
    parts = get_level_parts(mh,i)
    if i_am_in(parts)
      levels[i] = MultiFieldFESpace(map(sh -> sh[i], spaces);kwargs...)
    end
  end
  FESpaceHierarchy(mh,levels)
end

# Computing system matrices

function compute_hierarchy_matrices(trials::FESpaceHierarchy,
                                    tests::FESpaceHierarchy,
                                    a::Function,
                                    l::Function,
                                    qdegree::Integer)
  return compute_hierarchy_matrices(trials,tests,a,l,Fill(qdegree,num_levels(trials)))
end

function compute_hierarchy_matrices(trials::FESpaceHierarchy,
                                    tests::FESpaceHierarchy,
                                    a::Function,
                                    l::Function,
                                    qdegree::AbstractArray{<:Integer})
  nlevs = num_levels(trials)
  mh    = trials.mh

  @check length(qdegree) == nlevs

  A = nothing
  b = nothing
  mats = Vector{PSparseMatrix}(undef,nlevs)
  for lev in 1:nlevs
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      model = get_model(mh,lev)
      U = get_fe_space(trials,lev)
      V = get_fe_space(tests,lev)
      Ω = Triangulation(model)
      dΩ = Measure(Ω,qdegree[lev])
      ai(u,v) = a(u,v,dΩ)
      if lev == 1
        li(v) = l(v,dΩ)
        op    = AffineFEOperator(ai,li,U,V)
        A, b  = get_matrix(op), get_vector(op)
        mats[lev] = A
      else
        mats[lev] = assemble_matrix(ai,U,V)
      end
    end
  end
  return mats, A, b
end
