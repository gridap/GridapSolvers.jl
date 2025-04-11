module RefinementToolsTests
using MPI
using PartitionedArrays
using Gridap, Gridap.Algebra
using GridapDistributed
using Test

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.LinearSolvers

function get_model_hierarchy(parts,Dc,np_per_level)
  domain = ifelse(Dc == 2, (0,1,0,1), (0,1,0,1,0,1))
  nc = ifelse(Dc == 2, (4,4), (4,4,4))
  mh = CartesianModelHierarchy(parts,np_per_level,domain,nc)
  return mh
end

function main_driver(parts,mh)
  # FE Spaces
  order  = 2
  sol(x) = x[1]^2 + x[2]^2 - 3.0*x[1]*x[2]
  reffe  = ReferenceFE(lagrangian,Float64,order)
  tests  = TestFESpace(mh,reffe;conformity=:H1,dirichlet_tags="boundary")
  trials = TrialFESpace(tests,sol)
  solver = CGSolver(JacobiLinearSolver();verbose=i_am_main(parts),rtol=1.0e-8)

  nlevs = num_levels(mh)
  quad_order = 2*order+1
  for lev in 1:nlevs-1
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
      oph = AffineFEOperator(ah,lh,Uh,Vh)
      Ah  = get_matrix(oph)
      bh  = get_vector(oph)

      xh = pfill(0.0,partition(axes(Ah,2)))
      ns = numerical_setup(symbolic_setup(solver,Ah),Ah)
      solve!(xh,ns,bh)
      uH_projected = FEFunction(Uh,xh)

      _eh = uh-uH_projected
      eh  = sum(∫(_eh⋅_eh)*dΩh)
      i_am_main(parts) && println("Error H2h: ", eh)
      @test eh < 1.0e-10

      # Fine FEFunction -> Coarse FEFunction, by projection
      aH(u,v) = ∫(v⋅u)*dΩH
      lH(v)   = ∫(v⋅uH_projected)*dΩhH
      opH = AffineFEOperator(aH,lH,UH,VH)
      AH  = get_matrix(opH)
      bH  = get_vector(opH)

      xH = pfill(0.0,partition(axes(AH,2)))
      ns = numerical_setup(symbolic_setup(solver,AH),AH)
      solve!(xH,ns,bH)
      uh_projected = FEFunction(UH,xH)

      _eH = uH-uh_projected
      eH  = sum(∫(_eH⋅_eH)*dΩH)
      i_am_main(parts) && println("Error h2H: ", eH)
      @test eh < 1.0e-10

      # Coarse FEFunction -> Fine FEFunction, by interpolation
      uH_i = interpolate(uH,Uh)

      _eh = uH_i-uh
      eh  = sum(∫(_eh⋅_eh)*dΩh)
      i_am_main(parts) && println("Error h2H: ", eh)
      @test eh < 1.0e-10
    end
  end
end

function main(distribute,np,Dc,np_x_level)
  parts = distribute(LinearIndices((np,)))
  mh = get_model_hierarchy(parts,Dc,np_x_level)
  main_driver(parts,mh)
end

end