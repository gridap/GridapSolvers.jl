module GridapP4estExt

using Gridap
using GridapDistributed
using PartitionedArrays
using GridapSolvers

using GridapSolvers.MultilevelTools

using GridapP4est

"""
    P4estCartesianModelHierarchy(
      ranks,np_per_level,domain,nc;
      num_refs_coarse::Integer = 0,
      add_labels!::Function = (labels -> nothing),
      map::Function = identity,
      isperiodic = Tuple(fill(false,D))
    ) where D
  
  Returns a `ModelHierarchy` with a Cartesian model as coarsest level, using GridapP4est.jl. 
  The i-th level will be distributed among `np_per_level[i]` processors. 
  The seed model is given by `cmodel = CartesianDiscreteModel(domain,nc)`.
"""
function GridapSolvers.P4estCartesianModelHierarchy(
  ranks,np_per_level,domain,nc;
  num_refs_coarse = 0,
  add_labels! = (labels -> nothing),
  map = identity,
  isperiodic = Tuple(fill(false,length(nc)))
)
  cparts = generate_subparts(ranks,np_per_level[end])
  cmodel = CartesianDiscreteModel(domain,nc;map,isperiodic)
  add_labels!(get_face_labeling(cmodel))

  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(ranks,coarse_model,np_per_level)
  return mh
end

function GridapDistributed.DistributedAdaptedDiscreteModel(
  model  :: GridapP4est.OctreeDistributedDiscreteModel,
  parent :: GridapDistributed.DistributedDiscreteModel,
  glue   :: AbstractArray{<:Gridap.Adaptivity.AdaptivityGlue};
)
  GridapDistributed.DistributedAdaptedDiscreteModel(
    model.dmodel,parent,glue
  )
end

function GridapDistributed.DistributedAdaptedDiscreteModel(
  model  :: GridapP4est.OctreeDistributedDiscreteModel,
  parent :: GridapP4est.OctreeDistributedDiscreteModel,
  glue   :: AbstractArray{<:Gridap.Adaptivity.AdaptivityGlue};
)
  GridapDistributed.DistributedAdaptedDiscreteModel(
    model.dmodel,parent.dmodel,glue
  )
end

end # module