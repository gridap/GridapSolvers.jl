module DistributedGridTransferOperatorsTests
using MPI
using PartitionedArrays
using Gridap
using GridapDistributed
using GridapP4est
using Test

using GridapSolvers
using GridapSolvers.MultilevelTools

function get_model_hierarchy(parts,Dc,num_parts_x_level)
  mh = GridapP4est.with(parts) do
    if Dc == 2
      domain = (0,1,0,1)
      nc = (2,2)
    else
      @assert Dc == 3
      domain = (0,1,0,1,0,1)
      nc = (2,2,2)
    end
    num_refs_coarse = 2
    num_levels   = length(num_parts_x_level)
    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,nc)
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
    return ModelHierarchy(parts,coarse_model,num_parts_x_level)
  end
  return mh
end

function main_driver(parts,mh)
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

  a(u,v,dΩ) = ∫(v⋅u)*dΩ
  l(v,dΩ)   = ∫(v⋅u)*dΩ
  mats, A, b = compute_hierarchy_matrices(trials,a,l,qdegree)

  nlevs = num_levels(mh)
  for lev in 1:nlevs-1
    parts_h = get_level_parts(mh,lev)
    parts_H = get_level_parts(mh,lev+1)

    if i_am_in(parts_h)
      i_am_main(parts_h) && println("Lev : ", lev)
      Ah  = mats[lev]
      xh  = pfill(1.0,partition(axes(Ah,2)))
      yh1 = pfill(0.0,partition(axes(Ah,2)))
      yh2 = pfill(0.0,partition(axes(Ah,2)))
      yh3 = pfill(0.0,partition(axes(Ah,2)))

      if i_am_in(parts_H)
        AH  = mats[lev+1]
        xH  = pfill(1.0,partition(axes(AH,2)))
        yH1 = pfill(0.0,partition(axes(AH,2)))
        yH2 = pfill(0.0,partition(axes(AH,2)))
        yH3 = pfill(0.0,partition(axes(AH,2)))
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
        y_ref = pfill(1.0,partition(axes(AH,2)))
        tests = map(own_values(y_ref),own_values(yH1),own_values(yH2),own_values(yH3)) do y_ref,y1,y2,y3
          map(y -> norm(y-y_ref) < 1.e-3 ,[y1,y2,y3])
        end
        @test all(PartitionedArrays.getany(tests))
      end

      # ----    Prolongation    ----
      i_am_main(parts_h) && println("  > Prolongation")
      P1 = prolongations1[lev]
      mul!(yh1,P1,xH)

      P2 = prolongations2[lev]
      mul!(yh2,P2,xH)

      P3 = prolongations3[lev]
      mul!(yh3,P3,xH)

      y_ref = pfill(1.0,partition(axes(Ah,2)))
      tests = map(own_values(y_ref),own_values(yh1),own_values(yh2),own_values(yh3)) do y_ref,y1,y2,y3
        map(y -> norm(y-y_ref) < 1.e-3 ,[y1,y2,y3])
      end
      @test all(PartitionedArrays.getany(tests))
    end
  end
end

function main(distribute,np,Dc,np_x_level)
  parts = distribute(LinearIndices((np,)))
  mh = get_model_hierarchy(parts,Dc,np_x_level)
  main_driver(parts,mh)
end

end # module DistributedGridTransferOperatorsTests