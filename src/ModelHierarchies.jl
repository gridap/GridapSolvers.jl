
"""
  Single level for a ModelHierarchy.

  In each level, `cmodel` is the coarse model which is 
  first redistributed to obtain `cmodel_red` and then 
  refined to obtain `fmodel_red`. 

  Note that `model_red` and `red_glue` might be of type `Nothing`
  whenever there is no redistribution in a given level.
  
"""
struct ModelHierarchyLevel{A,B,C}
  level     :: Int
  model     :: A
  model_red :: B
  red_glue  :: C
end

"""
"""
struct ModelHierarchy
  level_parts :: Vector{PartitionedArrays.AbstractPData}
  levels      :: Vector{ModelHierarchyLevel}
end

num_levels(a::ModelHierarchy) = length(a.levels)
get_level(a::ModelHierarchy,level::Integer) = a.levels[level]

get_model(a::ModelHierarchy,level::Integer) = get_model(get_level(a,level))
get_model(a::ModelHierarchyLevel{A,Nothing}) where {A} = a.model
get_model(a::ModelHierarchyLevel{A,B}) where {A,B} = a.model_red

get_model_before_redist(a::ModelHierarchy,level::Integer) = get_model_before_redist(get_level(a,level))
get_model_before_redist(a::ModelHierarchyLevel) = a.model

has_redistribution(a::ModelHierarchy,level::Integer) = has_redistribution(a.levels[level])
has_redistribution(a::ModelHierarchyLevel{A,B,C}) where {A,B,C} = true
has_redistribution(a::ModelHierarchyLevel{A,B,Nothing}) where {A,B} = false

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
  meshes[num_levels] = ModelHierarchyLevel(num_levels,coarsest_model,nothing,nothing)

  for i = num_levels-1:-1:1
    modelH = get_model(meshes[i+1])
    if (num_procs_x_level[i]!=num_procs_x_level[i+1])
      # meshes[i+1].model is distributed among P processors
      # model_ref is distributed among Q processors, with P!=Q
      model_ref = Gridap.Refinement.refine(modelH,level_parts[i])
      model_red,red_glue = redistribute(model_ref)
    else
      model_ref = Gridap.Refinement.refine(modelH)
      model_red,red_glue = nothing,nothing
    end
    meshes[i] = ModelHierarchyLevel(i,model_ref,model_red,red_glue)
  end

  return ModelHierarchy(level_parts,meshes)
end