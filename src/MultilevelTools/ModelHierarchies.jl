
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
    const ModelHierarchy = HierarchicalArray{<:ModelHierarchyLevel}
  
  A `ModelHierarchy` is a hierarchical array of `ModelHierarchyLevel` objects. It stores the 
  adapted/redistributed models and the corresponding subcommunicators.

  For convenience, implements some of the API of `DiscreteModel`.
"""
const ModelHierarchy = HierarchicalArray{<:ModelHierarchyLevel}

get_model(a::ModelHierarchy,level::Integer) = get_model(a[level])
get_model(a::ModelHierarchyLevel{A,B,Nothing}) where {A,B} = a.model
get_model(a::ModelHierarchyLevel{A,B,C}) where {A,B,C} = a.model_red

get_model_before_redist(a::ModelHierarchy,level::Integer) = get_model_before_redist(a[level])
get_model_before_redist(a::ModelHierarchyLevel) = a.model

has_redistribution(a::ModelHierarchy,level::Integer) = has_redistribution(a[level])
has_redistribution(a::ModelHierarchyLevel{A,B,C,D}) where {A,B,C,D} = true
has_redistribution(a::ModelHierarchyLevel{A,B,C,Nothing}) where {A,B,C} = false

has_refinement(a::ModelHierarchy,level::Integer) = has_refinement(a[level])
has_refinement(a::ModelHierarchyLevel{A,B}) where {A,B} = true
has_refinement(a::ModelHierarchyLevel{A,Nothing}) where A = false

"""
    CartesianModelHierarchy(
      ranks::AbstractVector{<:Integer},
      np_per_level::Vector{<:NTuple{D,<:Integer}},
      domain::Tuple,
      nc::NTuple{D,<:Integer};
      nrefs::Union{<:Integer,Vector{<:Integer},Vector{<:NTuple{D,<:Integer}},NTuple{D,<:Integer}},
      map::Function = identity,
      isperiodic::NTuple{D,Bool} = Tuple(fill(false,D)),
      add_labels!::Function = (labels -> nothing),
    ) where D
  
  Returns a `ModelHierarchy` with a Cartesian model as coarsest level. The i-th level 
  will be distributed among `np_per_level[i]` processors. Two consecutive levels are 
  refined by a factor of `nrefs[i]`.

  ## Parameters: 

  - `ranks`: Initial communicator. Will be used to generate subcommunicators.
  - `domain`: Tuple containing the domain limits.
  - `nc`: Tuple containing the number of cells in each direction for the coarsest model.
  - `np_per_level`: Vector containing the number of processors we want to distribute
    each level into. Requires a tuple `np = (np_1,...,np_d)` for each level, then each 
    level will be distributed among `prod(np)` processors with `np_i` processors in the
    i-th direction.
  - `nrefs`: Vector containing the refinement factor for each level. Has `nlevs-1` entries, 
      and each entry can either be an integer (homogeneous refinement) or a tuple 
      with `D` integers (inhomogeneous refinement).
"""
function CartesianModelHierarchy(
  ranks::AbstractVector{<:Integer},
  np_per_level::Vector{<:NTuple{D,<:Integer}},
  domain::Tuple,
  nc::NTuple{D,<:Integer};
  nrefs::Union{<:Integer,Vector{<:Integer},Vector{<:NTuple{D,<:Integer}},NTuple{D,<:Integer}}=2,
  map::Function = identity,
  isperiodic::NTuple{D,Bool} = Tuple(fill(false,D)),
  add_labels!::Function = (labels -> nothing),
) where D

  lnr(x::Integer,n) = fill(Tuple(fill(x,D)),n)
  lnr(x::Vector{<:Integer},n) = [Tuple(fill(xi,D)) for xi in x]
  lnr(x::NTuple{D,<:Integer},n) = fill(x,n)
  lnr(x::Vector{<:NTuple{D,<:Integer}},n) = x

  nlevs = length(np_per_level)
  level_parts = generate_level_parts(ranks,Base.map(prod,np_per_level))
  level_nrefs = lnr(nrefs,length(level_parts)-1)
  @assert length(level_parts)-1 == length(level_nrefs)

  # Cartesian descriptors for each level
  level_descs = Vector{GridapDistributed.DistributedCartesianDescriptor}(undef,nlevs)
  ncells = nc
  for lev in nlevs:-1:1
    level_descs[lev] = GridapDistributed.DistributedCartesianDescriptor(
      level_parts[lev],np_per_level[lev],CartesianDescriptor(domain,ncells;map,isperiodic)
    )
    if lev != 1
      ncells = ncells .* level_nrefs[lev-1]
    end
  end
  map_main(ranks) do r
    @info "$(nlevs)-level CartesianModelHierarchy:\n$(join(["  > Level $(lev): "*repr("text/plain",level_descs[lev]) for lev in 1:nlevs],"\n"))"
  end
  
  # Coarsest level
  meshes = Vector{ModelHierarchyLevel}(undef,nlevs)
  if i_am_in(level_parts[nlevs]) 
    coarsest_model = CartesianDiscreteModel(level_parts[nlevs],np_per_level[nlevs],domain,nc;map,isperiodic)
    Base.map(add_labels!,local_views(get_face_labeling(coarsest_model)))
  else
    coarsest_model = nothing
  end
  meshes[nlevs] = ModelHierarchyLevel(nlevs,coarsest_model,nothing,nothing,nothing)

  # Remaining levels
  for lev in nlevs-1:-1:1
    fparts = level_parts[lev]
    cparts = level_parts[lev+1]
    # Refinement
    if i_am_in(cparts)
      cmodel = get_model(meshes[lev+1])
      model_ref  = Gridap.Adaptivity.refine(cmodel,level_nrefs[lev])
      ref_glue   = Gridap.Adaptivity.get_adaptivity_glue(model_ref)
    else
      model_ref, ref_glue = nothing, nothing, nothing
    end
    # Redistribution (if needed)
    if i_am_in(fparts) && (cparts !== fparts) 
      _model_ref = i_am_in(cparts) ? Gridap.Adaptivity.get_model(model_ref) : nothing
      model_red, red_glue = GridapDistributed.redistribute(_model_ref,level_descs[lev];old_ranks=cparts)
    else
      model_red, red_glue = nothing, nothing
    end
    meshes[lev] = ModelHierarchyLevel(lev,model_ref,ref_glue,model_red,red_glue)
  end
  return HierarchicalArray(meshes,level_parts)
end

"""
    P4estCartesianModelHierarchy(
      ranks,np_per_level,domain,nc::NTuple{D,<:Integer};
      num_refs_coarse::Integer = 0,
      add_labels!::Function = (labels -> nothing),
      map::Function = identity,
      isperiodic::NTuple{D,Bool} = Tuple(fill(false,D))
    ) where D
  
  Returns a `ModelHierarchy` with a Cartesian model as coarsest level, using GridapP4est.jl. 
  The i-th level will be distributed among `np_per_level[i]` processors. 
  The seed model is given by `cmodel = CartesianDiscreteModel(domain,nc)`.
"""
function P4estCartesianModelHierarchy(
  ranks,np_per_level,domain,nc::NTuple{D,<:Integer};
  num_refs_coarse::Integer = 0,
  add_labels!::Function = (labels -> nothing),
  map::Function = identity,
  isperiodic::NTuple{D,Bool} = Tuple(fill(false,D))
) where D
  cparts = generate_subparts(ranks,np_per_level[end])
  cmodel = CartesianDiscreteModel(domain,nc;map,isperiodic)
  add_labels!(get_face_labeling(cmodel))

  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(ranks,coarse_model,np_per_level)
  return mh
end

"""
    ModelHierarchy(root_parts,model,num_procs_x_level)

  - `root_parts`: Initial communicator. Will be used to generate subcommunicators.
  - `model`: Initial refinable distributed model. Will be set as coarsest level. 
  - `num_procs_x_level`: Vector containing the number of processors we want to distribute
      each level into. We need `num_procs_x_level[end]` to be equal to 
      the number of parts of `model`, and `num_procs_x_level[1]` to lower than the total 
      number of available processors in `root_parts`.
"""
function ModelHierarchy(
  root_parts        ::AbstractArray,
  model             ::GridapDistributed.DistributedDiscreteModel,
  num_procs_x_level ::Vector{<:Integer};
  mesh_refinement = true,
  kwargs...
)
  # Request correct number of parts from MAIN
  model_parts  = get_parts(model)
  my_num_parts = map(root_parts) do _p
    num_parts(model_parts) # == -1 if !i_am_in(my_parts)
  end
  main_num_parts = PartitionedArrays.getany(emit(my_num_parts))

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

function _model_hierarchy_without_refinement_bottom_up(
  root_parts::AbstractArray{T},
  bottom_model::GridapDistributed.DistributedDiscreteModel,
  num_procs_x_level::Vector{<:Integer}
) where T
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{Union{typeof(root_parts),GridapDistributed.MPIVoidVector{T}}}(undef,num_levels)
  meshes      = Vector{ModelHierarchyLevel}(undef,num_levels)

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

  mh = HierarchicalArray(meshes,level_parts)
  return mh
end

function _model_hierarchy_without_refinement_top_down(
  root_parts::AbstractArray{T},
  top_model::GridapDistributed.DistributedDiscreteModel,
  num_procs_x_level::Vector{<:Integer}
) where T
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{Union{typeof(root_parts),GridapDistributed.MPIVoidVector{T}}}(undef,num_levels)
  meshes      = Vector{ModelHierarchyLevel}(undef,num_levels)

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

  mh = HierarchicalArray(meshes,level_parts)
  return mh
end

function _model_hierarchy_by_refinement(
  root_parts::AbstractArray{T},
  coarsest_model::GridapDistributed.DistributedDiscreteModel,
  num_procs_x_level::Vector{<:Integer}; 
  num_refs_x_level=nothing
) where T
  # TODO: Implement support for num_refs_x_level? (future work)
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{Union{typeof(root_parts),GridapDistributed.MPIVoidVector{T}}}(undef,num_levels)
  meshes      = Vector{ModelHierarchyLevel}(undef,num_levels)

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

  mh = HierarchicalArray(meshes,level_parts)
  return convert_to_adapted_models(mh)
end

function _model_hierarchy_by_coarsening(
  root_parts::AbstractArray{T},
  finest_model::GridapDistributed.DistributedDiscreteModel,
  num_procs_x_level::Vector{<:Integer}; 
  num_refs_x_level=nothing
) where T
  # TODO: Implement support for num_refs_x_level? (future work)
  num_levels  = length(num_procs_x_level)
  level_parts = Vector{Union{typeof(root_parts),GridapDistributed.MPIVoidVector{T}}}(undef,num_levels)
  meshes      = Vector{ModelHierarchyLevel}(undef,num_levels)
  
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

  mh = HierarchicalArray(meshes,level_parts)
  return convert_to_adapted_models(mh)
end

function convert_to_adapted_models(mh::ModelHierarchy)
  map(linear_indices(mh),mh) do lev, mhl
    if lev == num_levels(mh)
      return mhl
    end

    if i_am_in(get_level_parts(mh,lev+1))
      model     = get_model_before_redist(mh,lev)
      parent    = get_model(mh,lev+1)
      ref_glue  = mhl.ref_glue
      model_ref = GridapDistributed.DistributedAdaptedDiscreteModel(model,parent,ref_glue)
    else
      model_ref = nothing
    end
    return ModelHierarchyLevel(lev,model_ref,mhl.ref_glue,mhl.model_red,mhl.red_glue)
  end
end