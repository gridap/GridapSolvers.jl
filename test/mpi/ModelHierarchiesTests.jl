module ModelHierarchiesTests

using MPI
using Gridap
using Gridap.FESpaces
using GridapDistributed
using PartitionedArrays
using GridapP4est

using GridapSolvers
using GridapSolvers.MultilevelTools

function model_hierarchy_free!(mh::ModelHierarchy)
  for lev in 1:num_levels(mh)
    model = get_model(mh,lev)
    isa(model,DistributedAdaptedDiscreteModel) && (model = model.model)
    octree_distributed_discrete_model_free!(model)
  end
end

function main(parts,num_parts_x_level,num_trees,num_refs_coarse)
  domain    = (0,1,0,1)
  cmodel    = CartesianDiscreteModel(domain,num_trees)

  num_levels   = length(num_parts_x_level)
  level_parts  = generate_level_parts(parts,num_parts_x_level)
  coarse_model = OctreeDistributedDiscreteModel(level_parts[num_levels],cmodel,num_refs_coarse)
  mh = ModelHierarchy(coarse_model,level_parts)

  sol(x) = x[1] + x[2]
  reffe  = ReferenceFE(lagrangian,Float64,1)
  tests  = TestFESpace(mh,reffe,conformity=:H1)
  trials = TrialFESpace(tests,sol)

  # model_hierarchy_free!(mh)
end

num_parts_x_level = [4,4,2,2] # Procs in each refinement level
num_trees = (1,1)             # Number of initial P4est trees
num_refs_coarse = 2           # Number of initial refinements

ranks = num_parts_x_level[1]
prun(main,mpi,ranks,num_parts_x_level,num_trees,num_refs_coarse)
MPI.Finalize()

end