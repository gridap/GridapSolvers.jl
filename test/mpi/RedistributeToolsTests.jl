module RedistributeToolsTests
  using MPI
  using PartitionedArrays
  using Gridap
  using GridapDistributed
  using GridapP4est
  using GridapSolvers
  using Test

  function model_hierarchy_free!(mh::ModelHierarchy)
    for lev in 1:num_levels(mh)
      model = get_model(mh,lev)
      isa(model,DistributedRefinedDiscreteModel) && (model = model.model)
      octree_distributed_discrete_model_free!(model)
    end
  end

  function run(parts,num_parts_x_level,num_trees,num_refs_coarse)
    domain       = (0,1,0,1)
    cmodel       = CartesianDiscreteModel(domain,num_trees)
    
    num_levels   = length(num_parts_x_level)
    level_parts  = GridapSolvers.generate_level_parts(parts,num_parts_x_level)
    coarse_model = OctreeDistributedDiscreteModel(level_parts[num_levels],cmodel,num_refs_coarse)
    mh = ModelHierarchy(coarse_model,level_parts)

    old_parts = level_parts[2]
    new_parts = level_parts[1]

    # FE Spaces
    order = 1
    u(x)  = x[1] + x[2]
    reffe = ReferenceFE(lagrangian,Float64,order)
    glue  = mh.levels[1].red_glue

    model_old = get_model_before_redist(mh.levels[1])
    VOLD  = TestFESpace(model_old,reffe,dirichlet_tags="boundary")
    UOLD  = TrialFESpace(VOLD,u)

    model_new = get_model(mh.levels[1])
    VNEW  = TestFESpace(model_new,reffe,dirichlet_tags="boundary")
    UNEW  = TrialFESpace(VNEW,u)

    # Triangulations
    qdegree = 2*order+1
    Ω_new   = Triangulation(model_new)
    dΩ_new  = Measure(Ω_new,qdegree)
    uh_new  = interpolate(u,UNEW)

    if GridapP4est.i_am_in(old_parts)
      Ω_old   = Triangulation(model_old)
      dΩ_old  = Measure(Ω_old,qdegree)
      uh_old  = interpolate(u,UOLD)
    else
      Ω_old   = nothing
      dΩ_old  = nothing
      uh_old  = nothing
    end

    # Old -> New
    uh_old_red = GridapSolvers.redistribute_fe_function(uh_old,
                                                       UNEW,
                                                       model_new,
                                                       glue)
    n = sum(∫(uh_old_red)*dΩ_new)
    if GridapP4est.i_am_in(old_parts)
      o = sum(∫(uh_old)*dΩ_old)
      @test o ≈ n
    end

    # New -> Old
    uh_new_red = GridapSolvers.redistribute_fe_function(uh_new,
                                                   UOLD,
                                                   model_old,
                                                   glue;
                                                   reverse=true)
    n = sum(∫(uh_new)*dΩ_new)
    if GridapP4est.i_am_in(old_parts)
      o = sum(∫(uh_new_red)*dΩ_old)
      @test o ≈ n
    end

    #model_hierarchy_free!(mh)
  end


  num_parts_x_level = [4,2,2]   # Procs in each refinement level
  num_trees         = (1,1)     # Number of initial P4est trees
  num_refs_coarse   = 2         # Number of initial refinements
  
  ranks = num_parts_x_level[1]
  prun(run,mpi,ranks,num_parts_x_level,num_trees,num_refs_coarse)
  MPI.Finalize()
end
