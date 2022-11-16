module OctreeDistributedDiscreteModelsTests
  using MPI
  using Test
  using Gridap
  using Gridap.ReferenceFEs
  using Gridap.FESpaces
  using PartitionedArrays
  using GridapDistributed
  using GridapP4est
  using GridapSolvers

  function run(parts,subdomains,num_parts_x_level)
    if length(subdomains) == 2
      domain=(0,1,0,1)
    else
      @assert length(subdomains) == 3
      domain=(0,1,0,1,0,1)
    end

    # Generate model
    level_parts  = GridapP4est.generate_level_parts(parts,num_parts_x_level)
    coarse_model = CartesianDiscreteModel(domain,subdomains)
    model        = GridapP4est.OctreeDistributedDiscreteModel(level_parts[2],coarse_model,1)

    # Refining and distributing
    fmodel , rglue  = refine(model,level_parts[1])
    dfmodel, dglue  = redistribute(fmodel)

    map_parts(GridapDistributed.local_views(fmodel)) do model
      println(num_cells(model))
      println(typeof(model))
    end

    # FESpaces tests
    sol(x) = x[1] + x[2]
    reffe  = ReferenceFE(lagrangian,Float64,1)
    Vh     = TestFESpace(fmodel, reffe; conformity=:H1)
    Uh     = TrialFESpace(sol,Vh)
    Ω      = Triangulation(fmodel)
    dΩ     = Measure(Ω,3)
    
    a(u,v) = ∫(v⋅u)*dΩ
    assemble_matrix(a,Uh,Vh)
  end

  #prun(run,mpi,4,(2,2),[4,2])
  #MPI.Finalize()
end # module
