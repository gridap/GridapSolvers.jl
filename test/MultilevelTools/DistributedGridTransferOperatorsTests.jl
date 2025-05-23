module DistributedGridTransferOperatorsTests
using MPI
using PartitionedArrays
using Gridap, Gridap.Algebra
using GridapDistributed
using Test

using GridapSolvers
using GridapSolvers.MultilevelTools

using GridapDistributed: change_ghost

function get_model_hierarchy(parts,Dc,np_per_level)
  domain = ifelse(Dc == 2, (0,1,0,1), (0,1,0,1,0,1))
  nc = ifelse(Dc == 2, (4,4), (4,4,4))
  mh = CartesianModelHierarchy(parts,np_per_level,domain,nc)
  return mh
end

function get_hierarchy_matrices(trials,tests,a,l,qdegree)
  mats, vecs = map(trials,tests) do trials,tests
    U = get_fe_space(trials)
    V = get_fe_space(tests)
    Ω = get_triangulation(U)
    dΩ = Measure(Ω,qdegree)
    ai(u,v) = a(u,v,dΩ)
    li(v) = l(v,dΩ)
    op    = AffineFEOperator(ai,li,U,V)
    return get_matrix(op), get_vector(op)
  end |> tuple_of_arrays
  return mats, vecs
end

function main_driver(parts,mh,sol,trials,tests,mats,vecs,qdegree,rm,mode)
  # Create Operators:
  ops = setup_transfer_operators(trials, qdegree; restriction_method=rm, mode=mode)
  restrictions, prolongations = ops

  nlevs = num_levels(mh)
  for lev in 1:nlevs-1
    parts_h = get_level_parts(mh,lev)
    parts_H = get_level_parts(mh,lev+1)

    if i_am_in(parts_h)
      i_am_main(parts) && println("  >> Level: ", lev)
      Ah = mats[lev]
      bh = vecs[lev]
      Uh = get_fe_space(trials,lev)
      Vh = get_fe_space(tests,lev)
      uh_ref = interpolate(sol,Uh)
      xh_ref = change_ghost(get_free_dof_values(uh_ref),axes(Ah,2);make_consistent=true)
      rh_ref = similar(xh_ref); mul!(rh_ref,Ah,xh_ref); rh_ref .= bh .- rh_ref;
      yh = similar(xh_ref)
      if mode == :solution
        xh = copy(xh_ref)
        yh_ref = xh_ref
      else
        xh = copy(rh_ref)
        yh_ref = rh_ref
      end

      if i_am_in(parts_H)
        AH = mats[lev+1]
        bH = vecs[lev+1]
        UH = get_fe_space(trials,lev+1)
        VH = get_fe_space(tests,lev+1)
        uH_ref = interpolate(sol,UH)
        xH_ref = change_ghost(get_free_dof_values(uH_ref),axes(AH,2);make_consistent=true)
        rH_ref = similar(xH_ref); mul!(rH_ref,AH,xH_ref); rH_ref .= bH .- rH_ref;
        yH = similar(xH_ref)
        if mode == :solution
          xH = copy(xH_ref)
          yH_ref = xH_ref
        else
          xH = copy(rH_ref)
          yH_ref = rH_ref
        end
      else
        xH_ref = nothing
        xH     = nothing
        yH_ref = nothing
        yH     = nothing
      end

      # ----    Restriction    ----
      i_am_main(parts) && println("    >>> Restriction")
      R = restrictions[lev]
      mul!(yH,R,xh)

      if i_am_in(parts_H)
        errors = map(own_values(yH_ref),own_values(yH)) do y_ref,y
          e = norm(y-y_ref)
          i_am_main(parts) && println("      - Error = ", e)
          return e < 1.e-3
        end
        @test PartitionedArrays.getany(errors)
      end

      # ----    Prolongation    ----
      i_am_main(parts) && println("    >>> Prolongation")
      P = prolongations[lev]
      mul!(yh,P,xH)

      errors = map(own_values(yh_ref),own_values(yh)) do y_ref,y
        e = norm(y-y_ref)
        i_am_main(parts) && println("      - Error = ", e)
        return e < 1.e-3
      end
      @test PartitionedArrays.getany(errors)
    end
  end
end

u_hdiv(x)  = VectorValue([x[2]-x[1],x[1]-x[2]])
u_h1(x)    = x[1]+x[2]
#u_h1(x)   = x[1]*(1-x[1])*x[2]*(1-x[2])
#u_hdiv(x) = VectorValue([x[1]*(1.0-x[1]),-x[2]*(1.0-x[2])])

function main(distribute,np,Dc,np_x_level)
  parts = distribute(LinearIndices((np,)))
  mh = get_model_hierarchy(parts,Dc,np_x_level)

  conformities = [:h1,:hdiv]
  solutions = [u_h1,u_hdiv]
  for order in [1,2]
    reffes = [ReferenceFE(lagrangian,Float64,order),ReferenceFE(raviart_thomas,Float64,order)]
    for (conf,u,reffe) in zip(conformities,solutions,reffes)
      tests  = TestFESpace(mh,reffe;dirichlet_tags="boundary")
      trials = TrialFESpace(tests,u)
      for mode in [:solution]#,:residual]
        for rm in [:projection]#,:interpolation]
          qdegree = 2*order + 1
          fx = zero(u(VectorValue(0.0,0.0)))
          a(u,v,dΩ) = ∫(v⋅u)*dΩ
          l(v,dΩ)   = ∫(v⋅fx)*dΩ
          mats, vecs = get_hierarchy_matrices(trials,tests,a,l,qdegree)
          if i_am_main(parts)
            println(repeat("=",80))
            println("> Testing transfers for")
            println("  - order                = ", order)
            println("  - conformity           = ", conf)
            println("  - transfer_mode        = ", mode)
            println("  - restriction_method   = ", rm)
          end
          main_driver(parts,mh,u,trials,tests,mats,vecs,qdegree,rm,mode)
        end
      end
    end
  end
end

end # module DistributedGridTransferOperatorsTests