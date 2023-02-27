
"""
  Single level for a ModelHierarchy.

  Note that `model_red` and `red_glue` might be of type `Nothing`
  whenever there is no redistribution in a given level.

  `ref_glue` is of type `Nothing` on the coarsest level.
"""
struct ModelHierarchyLevel{A,B,C,D}
  level     :: Int
  model     :: A
  ref_glue  :: B
  model_red :: C
  red_glue  :: D
end

"""
"""
struct ModelHierarchy
  level_parts :: Vector{PartitionedArrays.AbstractPData}
  levels      :: Vector{ModelHierarchyLevel}
end

num_levels(a::ModelHierarchy) = length(a.levels)
get_level(a::ModelHierarchy,level::Integer) = a.levels[level]

get_level_parts(a::ModelHierarchy) = a.level_parts
get_level_parts(a::ModelHierarchy,level::Integer) = a.level_parts[level]

get_model(a::ModelHierarchy,level::Integer) = get_model(get_level(a,level))
get_model(a::ModelHierarchyLevel{A,B,Nothing}) where {A,B} = a.model
get_model(a::ModelHierarchyLevel{A,B,C}) where {A,B,C} = a.model_red

get_model_before_redist(a::ModelHierarchy,level::Integer) = get_model_before_redist(get_level(a,level))
get_model_before_redist(a::ModelHierarchyLevel) = a.model

has_redistribution(a::ModelHierarchy,level::Integer) = has_redistribution(a.levels[level])
has_redistribution(a::ModelHierarchyLevel{A,B,C,D}) where {A,B,C,D} = true
has_redistribution(a::ModelHierarchyLevel{A,B,C,Nothing}) where {A,B,C} = false

has_refinement(a::ModelHierarchy,level::Integer) = has_refinement(a.levels[level])
has_refinement(a::ModelHierarchyLevel{A,B}) where {A,B} = true
has_refinement(a::ModelHierarchyLevel{A,Nothing}) where A = false

"""
  ModelHierarchy(parts,model,num_procs_x_level;num_refs_x_level)
  - `model`: Initial refinable distributed model. Will be set as coarsest level. 
  - `num_procs_x_level`: Vector containing the number of processors we want to distribute
                         each level into. We need `num_procs_x_level[end]` to be equal to 
                         the number of parts of `model`.
"""
function ModelHierarchy(root_parts        ::AbstractPData,
                        model             ::GridapDistributed.DistributedDiscreteModel,
                        num_procs_x_level ::Vector{<:Integer};
                        mesh_refinement = true,
                        kwargs...)

  # Request correct number of parts from MAIN
  model_parts  = get_parts(model)
  my_num_parts = map_parts(root_parts) do _p
    num_parts(model_parts) # == -1 if !i_am_in(my_parts)
  end
  main_num_parts = get_main_part(my_num_parts)

  if main_num_parts == num_procs_x_level[end] # Coarsest model
    if mesh_refinement
      return _model_hierarchy_by_refinement(root_parts,model,num_procs_x_level;kwargs...)
    else
      return _model_hierarchy_without_refinement_bottom_up(root_parts,model,num_procs_x_level;kwargs...)
    end
  end
  if main_num_parts == num_procs_x_level[1]   # Finest model
    if mesh_refinement
      return _model_hierarchy_by_coarsening(root_parts,model,num_procs_x_level;kwargs...)
    else
      return _model_hierarchy_without_refinement_top_down(root_parts,model,num_procs_x_level;kwargs...)
    end
  end
  @error "Model parts do not correspond to coarsest or finest parts!"
end

function _model_hierarchy_without_refinement_bottom_up(root_parts::AbstractPData,
                                                       bottom_model::GridapDistributed.DistributedDiscreteModel,
                                                       num_procs_x_level::Vector{<:Integer})
  num_levels         = length(num_procs_x_level)
  level_parts        = Vector{typeof(root_parts)}(undef,num_levels)
  meshes             = Vector{ModelHierarchyLevel}(undef,num_levels)

  level_parts[num_levels] = get_parts(bottom_model)
  meshes[num_levels] = ModelHierarchyLevel(num_levels,bottom_model,nothing,nothing,nothing)

  for i = num_levels-1:-1:1
    model = get_model(meshes[i+1])
    if (num_procs_x_level[i] != num_procs_x_level[i+1])
      level_parts[i]     = generate_subparts(root_parts,num_procs_x_level[i])
      model_red,red_glue = GridapDistributed.redistribute(model,level_parts[i])
    else
      level_parts[i]     = level_parts[i+1]
      model_red,red_glue = nothing,nothing
    end
    meshes[i] = ModelHierarchyLevel(i,model,nothing,model_red,red_glue)
  end

  mh = ModelHierarchy(level_parts,meshes)
  return convert_to_void_models(mh)
end

function _model_hierarchy_without_refinement_top_down(root_parts::AbstractPData,
                                                      top_model::GridapDistributed.DistributedDiscreteModel,
                                                      num_procs_x_level::Vector{<:Integer})
  num_levels         = length(num_procs_x_level)
  level_parts        = Vector{typeof(root_parts)}(undef,num_levels)
  meshes             = Vector{ModelHierarchyLevel}(undef,num_levels)

  level_parts[1] = get_parts(top_model)
  model = top_model
  for i = 1:num_levels-1
    if (num_procs_x_level[i] != num_procs_x_level[i+1])
      level_parts[i+1]   = generate_subparts(root_parts,num_procs_x_level[i+1])
      model_red = model
      model,red_glue = GridapDistributed.redistribute(model_red,level_parts[i+1])
    else
      level_parts[i+1]   = level_parts[i]
      model_red,red_glue = nothing, nothing
    end
    meshes[i] = ModelHierarchyLevel(i,model,nothing,model_red,red_glue)
  end
  meshes[num_levels] = ModelHierarchyLevel(num_levels,model,nothing,nothing,nothing)

  mh = ModelHierarchy(level_parts,meshes)
  return convert_to_void_models(mh)
end

function _model_hierarchy_by_refinement(root_parts::AbstractPData,
                                        coarsest_model::GridapDistributed.DistributedDiscreteModel,
                                        num_procs_x_level::Vector{<:Integer}; 
                                        num_refs_x_level=nothing)
  # TODO: Implement support for num_refs_x_level? (future work)
  num_levels         = length(num_procs_x_level)
  level_parts        = Vector{typeof(root_parts)}(undef,num_levels)
  meshes             = Vector{ModelHierarchyLevel}(undef,num_levels)

  level_parts[num_levels] = get_parts(coarsest_model)
  meshes[num_levels] = ModelHierarchyLevel(num_levels,coarsest_model,nothing,nothing,nothing)

  for i = num_levels-1:-1:1
    modelH = get_model(meshes[i+1])
    if (num_procs_x_level[i] != num_procs_x_level[i+1])
      # meshes[i+1].model is distributed among P processors
      # model_ref is distributed among Q processors, with P!=Q
      level_parts[i]     = generate_subparts(root_parts,num_procs_x_level[i])
      model_ref,ref_glue = Gridap.Adaptivity.refine(modelH)
      model_red,red_glue = GridapDistributed.redistribute(model_ref,level_parts[i])
    else
      level_parts[i]     = level_parts[i+1]
      model_ref,ref_glue = Gridap.Adaptivity.refine(modelH)
      model_red,red_glue = nothing,nothing
    end
    meshes[i] = ModelHierarchyLevel(i,model_ref,ref_glue,model_red,red_glue)
  end

  mh = ModelHierarchy(level_parts,meshes)
  return convert_to_adapted_models(mh)
end

function _model_hierarchy_by_coarsening(root_parts::AbstractPData,
                                        finest_model::GridapDistributed.DistributedDiscreteModel,
                                        num_procs_x_level::Vector{<:Integer}; 
                                        num_refs_x_level=nothing)
  # TODO: Implement support for num_refs_x_level? (future work)
  num_levels         = length(num_procs_x_level)
  level_parts        = Vector{typeof(root_parts)}(undef,num_levels)
  meshes             = Vector{ModelHierarchyLevel}(undef,num_levels)
  
  level_parts[1] = get_parts(finest_model)
  model = finest_model
  for i = 1:num_levels-1
    if (num_procs_x_level[i] != num_procs_x_level[i+1])
      level_parts[i+1]   = generate_subparts(root_parts,num_procs_x_level[i+1])
      model_red          = model
      model_ref,red_glue = GridapDistributed.redistribute(model_red,level_parts[i+1])
      model_H  ,ref_glue = Gridap.Adaptivity.coarsen(model_ref)
    else
      level_parts[i+1]   = level_parts[i]
      model_red          = nothing
      model_ref,red_glue = model, nothing
      model_H  ,ref_glue = Gridap.Adaptivity.coarsen(model_ref)
    end
    model     = model_H
    meshes[i] = ModelHierarchyLevel(i,model_ref,ref_glue,model_red,red_glue)
  end

  meshes[num_levels] = ModelHierarchyLevel(num_levels,model,nothing,nothing,nothing)

  mh = ModelHierarchy(level_parts,meshes)
  return convert_to_adapted_models(mh)
end

function convert_to_adapted_models(mh::ModelHierarchy)
  nlevs  = num_levels(mh)
  levels = Vector{ModelHierarchyLevel}(undef,nlevs)
  for lev in 1:nlevs-1
    cparts = get_level_parts(mh,lev+1)
    if i_am_in(cparts)
      model       = get_model_before_redist(mh,lev)
      parent      = get_model(mh,lev+1)
      ref_glue    = mh.levels[lev].ref_glue
      model_ref   = GridapDistributed.DistributedAdaptedDiscreteModel(model,parent,ref_glue)
    else
      model     = get_model_before_redist(mh,lev)
      model_ref = VoidDistributedDiscreteModel(model)
    end
    levels[lev] = ModelHierarchyLevel(lev,model_ref,mh.levels[lev].ref_glue,mh.levels[lev].model_red,mh.levels[lev].red_glue)
  end
  levels[nlevs] = mh.levels[nlevs]

  return ModelHierarchy(mh.level_parts,levels)
end

function convert_to_void_models(mh::ModelHierarchy)
  nlevs  = num_levels(mh)
  levels = Vector{ModelHierarchyLevel}(undef,nlevs)
  for lev in 1:nlevs-1
    cparts = get_level_parts(mh,lev+1)
    if i_am_in(cparts)
      model_ref = get_model_before_redist(mh,lev)
    else
      model     = get_model_before_redist(mh,lev)
      model_ref = VoidDistributedDiscreteModel(model)
    end
    levels[lev] = ModelHierarchyLevel(lev,model_ref,mh.levels[lev].ref_glue,mh.levels[lev].model_red,mh.levels[lev].red_glue)
  end
  levels[nlevs] = mh.levels[nlevs]

  return ModelHierarchy(mh.level_parts,levels)
end
