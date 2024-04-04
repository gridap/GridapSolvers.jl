module NavierStokesGMGApplication

using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function get_patch_smoothers(mh,tests,biform,patch_decompositions,qdegree)
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),patch_decompositions,patch_spaces) do tests, PD, Ph
    Vh = get_fe_space(tests)
    Ω  = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    ap = (u,du,dv) -> biform(u,du,dv,dΩ)
    patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh;is_nonlinear=true)
    return RichardsonSmoother(patch_smoother,10,0.2)
  end
  return smoothers
end

function get_trilinear_form(mh_lev,triform,qdegree)
  model = get_model(mh_lev)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  return (u,du,dv) -> triform(u,du,dv,dΩ)
end

function get_mesh_hierarchy(parts,nc,np_per_level)
  Dc = length(nc)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  num_refs_coarse = 0#(Dc == 2) ? 1 : 0
  
  num_levels   = length(np_per_level)
  cparts       = generate_subparts(parts,np_per_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,nc)

  labels = get_face_labeling(cmodel)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"walls",[1,5,7])
  add_tag_from_tags!(labels,"right",[2,8])

  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

function main(distribute,np,nc)
  parts = distribute(LinearIndices((prod(np),)))

  # Geometry
  Dc = length(nc)
  mh = get_mesh_hierarchy(parts,nc,[np,1])
  model = get_model(mh,1)

  # FE spaces
  order = 2
  qdegree = 2*(order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  u_wall = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
  u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

  tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["walls","top"]);
  trials_u = TrialFESpace(tests_u,[u_wall,u_top]);
  U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
  Q = TestFESpace(model,reffe_p;conformity=:L2) 

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  # Weak formulation
  Re = 10.0
  ν = 1/Re
  α = 1.e2
  f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
  
  poly = (Dc==2) ? QUAD : HEX
  Π_Qh = LocalProjectionMap(poly,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
  graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v,dΩ) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,dv,dΩ) = ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  lap(u,v,dΩ) = ∫(ν*∇(v)⊙∇(u))dΩ
  rhs(v,dΩ) = ∫(v⋅f)dΩ

  jac_u(u,du,dv,dΩ) = lap(du,dv,dΩ) + dc(u,du,dv,dΩ) + graddiv(du,dv,dΩ)
  jac((u,p),(du,dp),(dv,dq),dΩ) = jac_u(u,du,dv,dΩ) - ∫(divergence(dv)*dp)dΩ - ∫(divergence(du)*dq)dΩ

  res_u(u,v,dΩ) = lap(u,v,dΩ) + c(u,v,dΩ) + graddiv(u,v,dΩ) - rhs(v,dΩ)
  res((u,p),(v,q),dΩ) = res_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ

  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  jac_h(x,dx,dy) = jac(x,dx,dy,dΩ)
  res_h(x,dy) = res(x,dy,dΩ)
  op = FEOperator(res_h,jac_h,X,Y)

  # GMG Solver for u
  biforms = map(mhl -> get_trilinear_form(mhl,jac_u,qdegree),mh)
  patch_decompositions = PatchDecomposition(mh)
  smoothers = get_patch_smoothers(
    mh,tests_u,jac_u,patch_decompositions,qdegree
  )
  restrictions = setup_restriction_operators(
    tests_u,qdegree;mode=:residual,solver=IS_ConjugateGradientSolver(;reltol=1.e-6)
  )
  prolongations = setup_patch_prolongation_operators(
    tests_u,patch_decompositions,jac_u,graddiv,qdegree;is_nonlinear=true
  )
  gmg = GMGLinearSolver(
    mh,trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=2,mode=:preconditioner,verbose=i_am_main(parts),is_nonlinear=true
  )

  # Solver
  solver_u = gmg
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  solver_p.log.depth = 2

  diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
  bblocks = map(CartesianIndices((2,2))) do I
    (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
  end
  coeffs = [1.0 1.0;
            0.0 1.0]  
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts))

  nlsolver = NewtonSolver(solver;maxiter=20,atol=1e-14,rtol=1.e-7,verbose=i_am_main(parts))
  xh = solve(nlsolver,op);
end

end # module