module RefinementToolsTests
using MPI
using PartitionedArrays
using Gridap
using GridapDistributed
using GridapP4est
using Test
using IterativeSolvers

using GridapSolvers
using GridapSolvers.MultilevelTools

function run(parts,num_parts_x_level,coarse_grid_partition,num_refs_coarse)
  GridapP4est.with(parts) do
    domain       = (0,1,0,1)
    num_levels   = length(num_parts_x_level)
    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
    mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    # FE Spaces
    order  = 1
    sol(x) = x[1] + x[2]
    reffe  = ReferenceFE(lagrangian,Float64,order)
    tests  = TestFESpace(mh,reffe;conformity=:H1,dirichlet_tags="boundary")
    trials = TrialFESpace(tests,sol)

    quad_order = 3*order+1
    for lev in 1:num_levels-1
      fparts = get_level_parts(mh,lev)
      cparts = get_level_parts(mh,lev+1)

      if i_am_in(cparts)
        model_h = get_model_before_redist(mh,lev)
        Vh  = get_fe_space_before_redist(tests,lev)
        Uh  = get_fe_space_before_redist(trials,lev)
        Ωh  = get_triangulation(model_h)
        dΩh = Measure(Ωh,quad_order)
        uh  = interpolate(sol,Uh)

        model_H = get_model(mh,lev+1)
        VH  = get_fe_space(tests,lev+1)
        UH  = get_fe_space(trials,lev+1)
        ΩH  = get_triangulation(model_H)
        dΩH = Measure(ΩH,quad_order)
        uH  = interpolate(sol,UH)
        dΩhH = Measure(ΩH,Ωh,quad_order)

        # Coarse FEFunction -> Fine FEFunction, by projection
        ah(u,v) = ∫(v⋅u)*dΩh
        lh(v)   = ∫(v⋅uH)*dΩh
        Ah = assemble_matrix(ah,Uh,Vh)
        bh = assemble_vector(lh,Vh)

        xh = PVector(0.0,Ah.cols)
        IterativeSolvers.cg!(xh,Ah,bh;verbose=i_am_main(parts),reltol=1.0e-08)
        uH_projected = FEFunction(Uh,xh)

        _eh = uh-uH_projected
        eh  = sum(∫(_eh⋅_eh)*dΩh)
        i_am_main(parts) && println("Error H2h: ", eh)

        # Fine FEFunction -> Coarse FEFunction, by projection
        aH(u,v) = ∫(v⋅u)*dΩH
        lH(v)   = ∫(v⋅uH_projected)*dΩhH
        AH = assemble_matrix(aH,UH,VH)
        bH = assemble_vector(lH,VH)

        xH = PVector(0.0,AH.cols)
        IterativeSolvers.cg!(xH,AH,bH;verbose=i_am_main(parts),reltol=1.0e-08)
        uh_projected = FEFunction(UH,xH)

        _eH = uH-uh_projected
        eH  = sum(∫(_eH⋅_eH)*dΩH)
        i_am_main(parts) && println("Error h2H: ", eH)
      end
    end
  end
end


num_parts_x_level = [4,2,2]   # Procs in each refinement level
num_trees         = (1,1)     # Number of initial P4est trees
num_refs_coarse   = 2         # Number of initial refinements

ranks = num_parts_x_level[1]
with_backend(run,MPIBackend(),ranks,num_parts_x_level,num_trees,num_refs_coarse)
MPI.Finalize()
end
