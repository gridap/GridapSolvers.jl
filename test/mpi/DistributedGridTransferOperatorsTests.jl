module DistributedGridTransferOperatorsTests
using MPI
using PartitionedArrays
using Gridap
using GridapDistributed
using GridapP4est
using Test

using GridapSolvers
using GridapSolvers.MultilevelTools

function model_hierarchy_free!(mh::ModelHierarchy)
  for lev in 1:num_levels(mh)
    model = get_model(mh,lev)
    isa(model,DistributedRefinedDiscreteModel) && (model = model.model)
    octree_distributed_discrete_model_free!(model)
  end
end

function run(parts,num_parts_x_level,coarse_grid_partition,num_refs_coarse)
  domain       = (0,1,0,1)
  num_levels   = length(num_parts_x_level)
  cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)

  # Create Operators: 
  order = 1
  u(x)  = x[1] + x[2]
  reffe = ReferenceFE(lagrangian,Float64,order)

  tests  = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials = TrialFESpace(tests,u)

  qdegree = order*2+1
  ops = setup_transfer_operators(trials, qdegree)
  restrictions, prolongations = ops

  a(u,v,dΩ) = ∫(∇(v)⋅∇(u))*dΩ
  l(v,dΩ)   = ∫(v⋅u)*dΩ
  mats, A, b = compute_hierarchy_matrices(trials,a,l,qdegree)

  for lev in 1:num_levels-1
    parts_h = get_level_parts(mh,lev)
    parts_H = get_level_parts(mh,lev+1)

    if GridapP4est.i_am_in(parts_h)
      GridapP4est.i_am_main(parts_h) && println("Lev : ", lev)
      Ah = mats[lev]
      xh = PVector(1.0,Ah.cols)
      yh = PVector(0.0,Ah.rows)

      if GridapP4est.i_am_in(parts_H)
        AH = mats[lev+1]
        xH = PVector(1.0,AH.cols)
        yH = PVector(0.0,AH.rows)
      else
        xH = nothing
        yH = nothing
      end

      GridapP4est.i_am_main(parts_h) && println("  > Restriction")
      R = restrictions[lev]
      mul!(yH,R,xh)

      GridapP4est.i_am_main(parts_h) && println("  > Prolongation")
      P = prolongations[lev]
      mul!(yh,P,xH)
    end
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
