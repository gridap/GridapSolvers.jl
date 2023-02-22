module DistributedGridTransferOperatorsTests
using MPI
using PartitionedArrays
using Gridap
using GridapDistributed
using GridapP4est
using Test

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

    # Create Operators: 
    order = 1
    u(x)  = 1.0
    reffe = ReferenceFE(lagrangian,Float64,order)

    tests  = TestFESpace(mh,reffe;dirichlet_tags="boundary")
    trials = TrialFESpace(tests,u)

    qdegree = order*2+1
    ops1 = setup_transfer_operators(trials, qdegree; restriction_method=:projection, mode=:solution)
    restrictions1, prolongations1 = ops1
    ops2 = setup_transfer_operators(trials, qdegree; restriction_method=:interpolation, mode=:solution)
    restrictions2, prolongations2 = ops2
    ops3 = setup_transfer_operators(trials, qdegree; restriction_method=:dof_mask, mode=:solution)
    restrictions3, prolongations3 = ops3

    a(u,v,dΩ) = ∫(∇(v)⋅∇(u))*dΩ
    l(v,dΩ)   = ∫(v⋅u)*dΩ
    mats, A, b = compute_hierarchy_matrices(trials,a,l,qdegree)

    for lev in 1:num_levels-1
      parts_h = get_level_parts(mh,lev)
      parts_H = get_level_parts(mh,lev+1)

      if i_am_in(parts_h)
        i_am_main(parts_h) && println("Lev : ", lev)
        Ah  = mats[lev]
        xh  = PVector(1.0,Ah.cols)
        yh1 = PVector(0.0,Ah.cols)
        yh2 = PVector(0.0,Ah.cols)
        yh3 = PVector(0.0,Ah.cols)

        if i_am_in(parts_H)
          AH  = mats[lev+1]
          xH  = PVector(1.0,AH.cols)
          yH1 = PVector(0.0,AH.cols)
          yH2 = PVector(0.0,AH.cols)
          yH3 = PVector(0.0,AH.cols)
        else
          xH  = nothing
          yH1 = nothing
          yH2 = nothing
          yH3 = nothing
        end

        # ----    Restriction    ----
        i_am_main(parts_h) && println("  > Restriction")
        R1 = restrictions1[lev]
        mul!(yH1,R1,xh)

        R2 = restrictions2[lev]
        mul!(yH2,R2,xh)

        R3 = restrictions3[lev]
        mul!(yH3,R3,xh)

        if i_am_in(parts_H)
          y_ref = PVector(1.0,AH.cols)
          tests = map_parts(y_ref.owned_values,yH1.owned_values,yH2.owned_values,yH3.owned_values) do y_ref,y1,y2,y3
            map(y -> norm(y-y_ref) < 1.e-3 ,[y1,y2,y3])
          end
          @test all(tests.part)
        end

        # ----    Prolongation    ----
        i_am_main(parts_h) && println("  > Prolongation")
        P1 = prolongations1[lev]
        mul!(yh1,P1,xH)

        P2 = prolongations2[lev]
        mul!(yh2,P2,xH)

        P3 = prolongations3[lev]
        mul!(yh3,P3,xH)

        y_ref = PVector(1.0,Ah.cols)
        tests = map_parts(y_ref.owned_values,yh1.owned_values,yh2.owned_values,yh2.owned_values) do y_ref,y1,y2,y3
          map(y -> norm(y-y_ref) < 1.e-3 ,[y1,y2,y3])
        end
        @test all(tests.part)

      end
    end
  end
end

num_parts_x_level = [4,2,2]   # Procs in each refinement level
num_trees         = (1,1)     # Number of initial P4est trees
num_refs_coarse   = 2         # Number of initial refinements

num_ranks = num_parts_x_level[1]
with_backend(run,MPIBackend(),num_ranks,num_parts_x_level,num_trees,num_refs_coarse)
println("AT THE END")
MPI.Finalize()
end
