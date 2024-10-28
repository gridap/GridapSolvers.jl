module GridapP4estExt

using Gridap
using GridapP4est

export P4estCartesianModelHierarchy

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

end # module