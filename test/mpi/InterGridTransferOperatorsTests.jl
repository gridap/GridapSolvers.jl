module InterGridTransferOperatorsTests
  """
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
    level_parts  = GridapSolvers.generate_level_parts(parts,num_parts_x_level)

    coarse_model = OctreeDistributedDiscreteModel(level_parts[num_levels],cmodel,num_refs_coarse)
    mh           = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    println(typeof(level_parts[1]))

    # FE Spaces
    println(" > Testing FESpaces")
    order  = 1
    reffe  = ReferenceFE(lagrangian,Float64,order)
    tests  = TestFESpace(mh,reffe,dirichlet_tags="boundary")
    trials = TrialFESpace(u,tests)

    # Transfer ops
    println(" > Testing operators")
    qdegree = 2
    for lev in 1:num_levels-1
      println("   > Level num ", lev)
      parts = get_level_parts(mh,lev)
      if GridapP4est.i_am_in(parts)
        R = RestrictionOperator(lev,trials,qdegree)
        P = ProlongationOperator(lev,trials,qdegree)
        @test isa(R,DistributedGridTransferOperator{Val{:restriction},Val{true}})
        @test isa(P,DistributedGridTransferOperator{Val{:prolongation},Val{true}})
      end
    end

    println(" > Testing setup_transfer_operators")
    ops = setup_transfer_operators(trials,qdegree)

    #model_hierarchy_free!(mh)
  end

  num_parts_x_level = [4,2,2]   # Procs in each refinement level
  num_trees         = (1,1)     # Number of initial P4est trees
  num_refs_coarse   = 2         # Number of initial refinements
  
  ranks = num_parts_x_level[1]
  prun(run,mpi,ranks,num_parts_x_level,num_trees,num_refs_coarse)
  MPI.Finalize()
  """
end
