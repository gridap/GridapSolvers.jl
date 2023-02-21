
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
    cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)

    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)

    fparts       = generate_subparts(parts,num_parts_x_level[1])
    fine_model   = OctreeDistributedDiscreteModel(fparts,cmodel,num_refs_coarse + num_levels)
    mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    # Create Operators: 
    order = 1
    u(x)  = 1.0
    reffe = ReferenceFE(lagrangian,Float64,order)

    tests  = TestFESpace(mh,reffe;dirichlet_tags="boundary")
    trials = TrialFESpace(tests,u)

    qdegree = order*2+1
    ops = setup_transfer_operators(trials, qdegree; restriction_method=:projection, mode=:solution)
    restrictions, prolongations = ops

    a(u,v,dΩ) = ∫(∇(v)⋅∇(u))*dΩ
    l(v,dΩ)   = ∫(v⋅u)*dΩ
    mats, A, b = compute_hierarchy_matrices(trials,a,l,qdegree)

    for lev in 1:num_levels-1
      parts_h = get_level_parts(mh,lev)
      parts_H = get_level_parts(mh,lev+1)

      if i_am_in(parts_h)
        i_am_main(parts_h) && println("Lev : ", lev)
        Ah = mats[lev]
        xh = PVector(1.0,Ah.cols)
        yh = PVector(0.0,Ah.cols)

        if i_am_in(parts_H)
          AH = mats[lev+1]
          xH = PVector(1.0,AH.cols)
          yH = PVector(0.0,AH.cols)
        else
          xH = nothing
          yH = nothing
        end

        i_am_main(parts_h) && println("  > Restriction")
        R = restrictions[lev]
        mul!(yH,R,xh)
        i_am_in(parts_H) && display(yH.values)

        i_am_main(parts_h) && println("  > Prolongation")
        P = prolongations[lev]
        mul!(yh,P,xH)
        i_am_in(parts_h) && display(yh.values)

      end
    end
  end
end

num_parts_x_level = [4,2,2]   # Procs in each refinement level
#num_parts_x_level = [1,1,1]   # Procs in each refinement level
num_trees         = (1,1)     # Number of initial P4est trees
num_refs_coarse   = 1         # Number of initial refinements

num_ranks = num_parts_x_level[1]
with_backend(run,MPIBackend(),num_ranks,num_parts_x_level,num_trees,num_refs_coarse)
println("AT THE END")
MPI.Finalize()




