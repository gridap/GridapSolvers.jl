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

function Gridap.FESpaces.TestFESpace(
      mh::ModelHierarchyLevel{A,B,C,Nothing},args...;kwargs...) where {A,B,C}
  Vh = TestFESpace(get_model(mh),args...;kwargs...)
  FESpaceHierarchyLevel(mh.level,Vh,nothing)
end

function Gridap.FESpaces.TestFESpace(
      mh::ModelHierarchyLevel{A,B,C,D},args...;kwargs...) where {A,B,C,D}
  Vh     = TestFESpace(get_model_before_redist(mh),args...;kwargs...)
  Vh_red = TestFESpace(get_model(mh),args...;kwargs...)
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
    parts = get_level_parts(mh,i)
    if (GridapP4est.i_am_in(parts))
       Vh = TestFESpace(get_level(mh,i),args...;kwargs...)
       test_spaces[i] = Vh
    end
  end
  FESpaceHierarchy(mh,test_spaces)
end

function Gridap.FESpaces.TrialFESpace(u,a::FESpaceHierarchy)
  trial_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(a.mh))
  for i=1:num_levels(a.mh)
    parts = get_level_parts(a.mh,i)
    if (GridapP4est.i_am_in(parts))
       Uh = TrialFESpace(u,a[i])
       trial_spaces[i] = Uh
    end
  end
  FESpaceHierarchy(a.mh,trial_spaces)
end

function Gridap.FESpaces.TrialFESpace(a::FESpaceHierarchy)
  trial_spaces = Vector{FESpaceHierarchyLevel}(undef,num_levels(a.mh))
  for i=1:num_levels(a.mh)
    parts = get_level_parts(a.mh,i)
    if (GridapP4est.i_am_in(parts))
       Uh = TrialFESpace(a[i])
       trial_spaces[i] = Uh
    end
  end
  FESpaceHierarchy(a.mh,trial_spaces)
end
