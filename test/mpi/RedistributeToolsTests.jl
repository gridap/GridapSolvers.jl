module RedistributeToolsTests
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
      octree_distributed_discrete_model_free!(model)
    end
  end

  function run(parts,num_parts_x_level,num_trees,num_refs_coarse)
    domain       = (0,1,0,1)
    cmodel       = CartesianDiscreteModel(domain,num_trees)
    coarse_model = OctreeDistributedDiscreteModel(parts,cmodel,num_refs_coarse)
    mh           = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    # FE Spaces
    order = 1
    reffe = ReferenceFE(lagrangian,Float64,order)
    glue  = mh.levels[1].red_glue

    model_old = get_model_before_redist(mh.levels[1])
    VOLD  = TestFESpace(model_old,reffe,dirichlet_tags="boundary")
    UOLD  = TrialFESpace(u,VOLD)

    model_new = get_model(mh.levels[1])
    VNEW  = TestFESpace(model_new,reffe,dirichlet_tags="boundary")
    UNEW  = TrialFESpace(u,VNEW)

    # Triangulations
    qdegree = 2*order+1
    Ω_old   = Triangulation(model_old)
    dΩ_old  = Measure(Ω_old,qdegree)
    Ω_new   = Triangulation(model_new)
    dΩ_new  = Measure(Ω_new,qdegree)

    # Old -> New
    uhold = interpolate(u,UOLD)
    uhnew = GridapSolvers.redistribute_fe_function(uhold,
                                                   UNEW,
                                                   model_new,
                                                   glue)
    o = sum(∫(uhold)*dΩ_old)
    n = sum(∫(uhnew)*dΩ_new)
    @test o ≈ n

    # New -> Old
    uhnew = interpolate(u,UNEW)
    uhold = GridapSolvers.redistribute_fe_function(uhnew,
                                                   UOLD,
                                                   model_old,
                                                   glue;
                                                   reverse=true)
    o = sum(∫(uhnew)*dΩ_new)
    n = sum(∫(uhold)*dΩ_old)
    @test o ≈ n

    model_hierarchy_free!(mh)
  end


  num_parts_x_level = [4,2,2]   # Procs in each refinement level
  num_trees         = (1,1)     # Number of initial P4est trees
  num_refs_coarse   = 2         # Number of initial refinements
  
  ranks = num_parts_x_level[1]
  prun(run,mpi,ranks,num_parts_x_level,num_trees,num_refs_coarse)
  MPI.Finalize()
end
