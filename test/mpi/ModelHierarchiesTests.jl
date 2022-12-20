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

function main(parts,num_parts_x_level)
  # Start from coarse, refine models
  domain       = (0,1,0,1)
  num_levels   = length(num_parts_x_level)
  cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,(2,2))
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,2)
  mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)

  sol(x) = x[1] + x[2]
  reffe  = ReferenceFE(lagrangian,Float64,1)
  tests  = TestFESpace(mh,reffe,conformity=:H1)
  trials = TrialFESpace(tests,sol)

  # Start from fine, coarsen models
  domain     = (0,1,0,1)
  fparts     = generate_subparts(parts,num_parts_x_level[1])
  fmodel     = CartesianDiscreteModel(domain,(2,2))
  fine_model = OctreeDistributedDiscreteModel(fparts,fmodel,8)
  mh = ModelHierarchy(parts,fine_model,num_parts_x_level)

  sol(x) = x[1] + x[2]
  reffe  = ReferenceFE(lagrangian,Float64,1)
  tests  = TestFESpace(mh,reffe,conformity=:H1)
  trials = TrialFESpace(tests,sol)

  # model_hierarchy_free!(mh)
end

num_parts_x_level = [4,4,2,2] # Procs in each refinement level

ranks = num_parts_x_level[1]
prun(main,mpi,ranks,num_parts_x_level)
MPI.Finalize()

end