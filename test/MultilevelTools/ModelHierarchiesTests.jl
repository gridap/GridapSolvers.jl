module ModelHierarchiesTests

using MPI
using Gridap
using Gridap.FESpaces, Gridap.Algebra
using GridapDistributed
using PartitionedArrays
using GridapP4est

using GridapSolvers
using GridapSolvers.MultilevelTools

function main(distribute,np,num_parts_x_level)
  parts = distribute(LinearIndices((prod(np),)))
  GridapP4est.with(parts) do
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
  end
end

end