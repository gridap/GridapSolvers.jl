module GridapP4estExtTests

using Test
using Gridap
using GridapDistributed
using PartitionedArrays
using GridapP4est

using GridapSolvers
using GridapSolvers.MultilevelTools

function test_mh(mh)
  sol(x) = x[1] + x[2]
  reffe  = ReferenceFE(lagrangian,Float64,1)
  tests  = TestFESpace(mh,reffe,conformity=:H1)
  trials = TrialFESpace(tests,sol)
  @test true
end

function main(distribute,np,np_per_level)
  domain = (0,1,0,1)
  parts = distribute(LinearIndices((prod(np),)))
  GridapP4est.with(parts) do
    # Start from coarse, refine models
    num_levels   = length(np_per_level)
    cparts       = generate_subparts(parts,np_per_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,(2,2))
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,2)
    mh = ModelHierarchy(parts,coarse_model,np_per_level)
    test_mh(mh)

    # Start from fine, coarsen models
    fparts     = generate_subparts(parts,np_per_level[1])
    fmodel     = CartesianDiscreteModel(domain,(2,2))
    fine_model = OctreeDistributedDiscreteModel(fparts,fmodel,8)
    mh = ModelHierarchy(parts,fine_model,np_per_level)
    test_mh(mh)

    # P4estCartesianModelHierarchy
    mh = P4estCartesianModelHierarchy(parts,np_per_level,domain,(2,2))
  end
end

with_mpi() do distribute
  main(distribute,4,[4,2,2,1])
end

end