module InterGridTransferOperatorsTests
  using MPI
  using PartitionedArrays
  using Gridap
  using GridapDistributed
  using GridapP4est
  using GridapSolvers
  using Test

  u(x) = x[1] + x[2]

  function model_hierarchy_free!(mh::ModelHierarchy)
    for lev in 1:num_levels(mh)
      model = get_model(mh,lev)
      isa(model,DistributedRefinedDiscreteModel) && (model = model.model)
      octree_distributed_discrete_model_free!(model)
    end
  end

  function run(parts,num_parts_x_level,num_trees,num_refs_coarse)
    num_levels   = length(num_parts_x_level)
    domain       = (0,1,0,1)
    cmodel       = CartesianDiscreteModel(domain,num_trees)
    coarse_model = OctreeDistributedDiscreteModel(parts,cmodel,num_refs_coarse)
    mh           = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    # FE Spaces
    order  = 1
    reffe  = ReferenceFE(lagrangian,Float64,order)
    tests  = TestFESpace(mh,reffe,dirichlet_tags="boundary")
    trials = TrialFESpace(u,tests)

    # Transfer ops
    qdegree = 2
    R = RestrictionOperator(1,trials,qdegree)
    P = ProlongationOperator(1,trials,qdegree)
    @test isa(R,DistributedGridTransferOperator{Val{:restriction},Val{true}})
    @test isa(P,DistributedGridTransferOperator{Val{:prolongation},Val{true}})

    R = RestrictionOperator(2,trials,qdegree)
    P = ProlongationOperator(2,trials,qdegree)
    @test isa(R,DistributedGridTransferOperator{Val{:restriction},Val{false}})
    @test isa(P,DistributedGridTransferOperator{Val{:prolongation},Val{false}})

    #ops = setup_transfer_operators(trials,qdegree)

    model_hierarchy_free!(mh)
  end


  num_parts_x_level = [4,2,2]   # Procs in each refinement level
  num_trees         = (1,1)     # Number of initial P4est trees
  num_refs_coarse   = 2         # Number of initial refinements
  
  ranks = num_parts_x_level[1]
  prun(run,mpi,ranks,num_parts_x_level,num_trees,num_refs_coarse)
  MPI.Finalize()
end
