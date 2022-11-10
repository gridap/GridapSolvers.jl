
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

get_level_parts(a::ModelHierarchy,level::Integer) = a.level_parts[level]

get_model(a::ModelHierarchy,level::Integer) = get_model(get_level(a,level))
get_model(a::ModelHierarchyLevel{A,B,Nothing}) where {A,B} = a.model
get_model(a::ModelHierarchyLevel{A,B,C}) where {A,B,C} = a.model_red

get_model_before_redist(a::ModelHierarchy,level::Integer) = get_model_before_redist(get_level(a,level))
get_model_before_redist(a::ModelHierarchyLevel) = a.model

has_redistribution(a::ModelHierarchy,level::Integer) = has_redistribution(a.levels[level])
has_redistribution(a::ModelHierarchyLevel{A,B,C,D}) where {A,B,C,D} = true
has_redistribution(a::ModelHierarchyLevel{A,B,C,Nothing}) where {A,B,C} = false

"""
  ModelHierarchy(parts,model,num_procs_x_level;num_refs_x_level)
  - `model`: Initial refinable distributed model. Will be set as coarsest level. 
  - `num_procs_x_level`: Vector containing the number of processors we want to distribute
                         each level into. We need `num_procs_x_level[end]` to be equal to 
                         the number of parts of `model`.
"""
function ModelHierarchy(parts,coarsest_model::GridapDistributed.AbstractDistributedDiscreteModel,num_procs_x_level::Vector{Int}; num_refs_x_level=nothing)
  # TODO: Implement support for num_refs_x_level? (future work)
  num_levels  = length(num_procs_x_level)
  level_parts = generate_level_parts(parts,num_procs_x_level)

  meshes = Vector{ModelHierarchyLevel}(undef,num_levels)
  meshes[num_levels] = ModelHierarchyLevel(num_levels,coarsest_model,nothing,nothing,nothing)

  for i = num_levels-1:-1:1
    modelH = get_model(meshes[i+1])
    if (num_procs_x_level[i] != num_procs_x_level[i+1])
      # meshes[i+1].model is distributed among P processors
      # model_ref is distributed among Q processors, with P!=Q
      model_ref,ref_glue = Gridap.Adaptivity.refine(modelH,level_parts[i])
      model_red,red_glue = redistribute(model_ref)
    else
      model_ref,ref_glue = Gridap.Adaptivity.refine(modelH)
      model_red,red_glue = nothing,nothing
    end
    meshes[i] = ModelHierarchyLevel(i,model_ref,ref_glue,model_red,red_glue)
  end

  mh = ModelHierarchy(level_parts,meshes)
  return convert_to_refined_models(mh)
end

function convert_to_refined_models(mh::ModelHierarchy)
  nlevs  = num_levels(mh)
  levels = Vector{ModelHierarchyLevel}(undef,nlevs)
  for lev in 1:nlevs-1
    model       = get_model_before_redist(mh,lev)
    parent      = get_model(mh,lev+1)
    ref_glue    = change_parts(mh.levels[lev].ref_glue,get_parts(model);default=void(AdaptivityGlue))
    model_ref   = DistributedAdaptedDiscreteModel(model,parent,ref_glue)

    levels[lev] = ModelHierarchyLevel(lev,model_ref,ref_glue,mh.levels[lev].model_red,mh.levels[lev].red_glue)
  end
  levels[nlevs] = mh.levels[nlevs]

  return ModelHierarchy(mh.level_parts,levels)
end
