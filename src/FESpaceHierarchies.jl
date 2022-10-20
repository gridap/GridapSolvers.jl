struct FESpaceHierarchyLevel{A,B}
  level        :: Int
  fe_space     :: A
  fe_space_red :: B
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

function get_space(a::FESpaceHierarchyLevel{A,Nothing}) where {A}
  a.fe_space
end

function get_space(a::FESpaceHierarchyLevel{A,B}) where {A,B}
  a.fe_space_red
end

function get_space_before_redist(a::FESpaceHierarchyLevel)
  a.fe_space
end

function Gridap.FESpaces.TestFESpace(
      mh::ModelHierarchyLevel{A,B,Nothing},args...;kwargs...) where {A,B}
  Vh = TestFESpace(mh.model,args...;kwargs...)
  FESpaceHierarchyLevel(mh.level,Vh,nothing)
end

function Gridap.FESpaces.TestFESpace(
      mh::ModelHierarchyLevel{A,B,C},args...;kwargs...) where {A,B,C}
  Vh     = TestFESpace(mh.model,args...;kwargs...)
  Vh_red = TestFESpace(mh.model_red,args...;kwargs...)
  FESpaceHierarchyLevel(mh.level,Vh,Vh_red)
end

function Gridap.FESpaces.TrialFESpace(u,a::FESpaceHierarchyLevel{A,Nothing}) where {A}
  Uh = TrialFESpace(u,a.fe_space)
  FESpaceHierarchyLevel(a.level,Uh,nothing)
end

function Gridap.FESpaces.TrialFESpace(u,a::FESpaceHierarchyLevel{A,B}) where {A,B}
  Uh     = TrialFESpace(u,a.fe_space)
  Uh_red = TrialFESpace(u,a.fe_space_red)
  FESpaceHierarchyLevel(a.level,Uh,Uh_red)
end

function Gridap.FESpaces.TestFESpace(mh::ModelHierarchy,args...;kwargs...) where {A,B}
  test_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(mh))
  for i=1:num_levels(mh)
    model = get_model(mh,i)
    if (GridapP4est.i_am_in(model.parts))
       Vh = TestFESpace(get_level(mh,i),args...;kwargs...)
       test_spaces[i]  = Vh
    end
  end
  FESpaceHierarchy(mh,test_spaces)
end

function Gridap.FESpaces.TrialFESpace(u,a::FESpaceHierarchy)
  trial_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(a.mh))
  for i=1:num_levels(a.mh)
    model = get_model(a.mh,i)
    if (GridapP4est.i_am_in(model.parts))
       Uh = TrialFESpace(u,a[i])
       trial_spaces[i]  = Uh
    end
  end
  FESpaceHierarchy(a.mh,trial_spaces)
end

function Gridap.FESpaces.TrialFESpace(a::FESpaceHierarchy)
  trial_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(a.mh))
  for i=1:num_levels(a.mh)
    model = get_model(a.mh,i)
    if (GridapP4est.i_am_in(model.parts))
       Uh = TrialFESpace(a[i])
       trial_spaces[i]  = Uh
    end
  end
  FESpaceHierarchy(a.mh,trial_spaces)
end
