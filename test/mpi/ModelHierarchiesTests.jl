module ModelHierarchiesTests

using MPI
using Gridap
using GridapDistributed
using PartitionedArrays
using GridapSolvers
using GridapP4est

function model_hierarchy_free!(mh::ModelHierarchy)
  for lev in 1:num_levels(mh)
    model = get_model(mh,lev)
    octree_distributed_discrete_model_free!(model)
  end
end

function main(parts,num_parts_x_level,num_trees,num_refs_coarse)
  domain    = (0,1,0,1)
  cmodel    = CartesianDiscreteModel(domain,num_trees)

  num_levels = length(num_parts_x_level)
  coarse_model = OctreeDistributedDiscreteModel(parts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)
  model_hierarchy_free!(mh)
end

num_parts_x_level = [4,4,2,2] # Procs in each refinement level
num_trees = (1,1)             # Number of initial P4est trees
num_refs_coarse = 2           # Number of initial refinements

ranks = num_parts_x_level[1]
prun(main,mpi,ranks,num_parts_x_level,num_trees,num_refs_coarse)
MPI.Finalize()

end