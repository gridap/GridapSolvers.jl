```@meta
EditURL = "../../../test/Applications/NavierStokesGMG.jl"
```

````@example NavierStokesGMG
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
using GridapSolvers.BlockSolvers: NonlinearSystemBlock, LinearSystemBlock, BiformBlock, BlockTriangularSolver

function get_patch_smoothers(mh,tests,biform,patch_decompositions,qdegree;is_nonlinear=false)
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  nlevs = num_levels(mh)
  smoothers = map(view(tests,1:nlevs-1),patch_decompositions,patch_spaces) do tests, PD, Ph
    Vh = get_fe_space(tests)
    Ω  = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    ap = (u,du,dv) -> biform(u,du,dv,dΩ)
    patch_smoother = PatchBasedLinearSolver(ap,Ph,Vh;is_nonlinear)
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

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"bottom",[1,2,5])
  add_tag_from_tags!(labels,"walls",[7,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
  add_tag_from_tags!(labels,"bottom",[1,2,3,4,9,10,13,14,21])
  add_tag_from_tags!(labels,"walls",[17,18,23,25,26])
end

function main(distribute,np,nc,np_per_level)
  parts = distribute(LinearIndices((prod(np),)))

  Dc = length(nc)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
  mh = CartesianModelHierarchy(parts,np_per_level,domain,nc;add_labels! = add_labels!)
  model = get_model(mh,1)

  order = 2
  qdegree = 2*(order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  u_bottom = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
  u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

  tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["bottom","top"]);
  trials_u = TrialFESpace(tests_u,[u_bottom,u_top]);
  U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
  Q = TestFESpace(model,reffe_p;conformity=:L2)

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

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

  biforms = map(mhl -> get_trilinear_form(mhl,jac_u,qdegree),mh)
  patch_decompositions = PatchDecomposition(mh)
  smoothers = get_patch_smoothers(
    mh,trials_u,jac_u,patch_decompositions,qdegree;is_nonlinear=true
  )
  prolongations = setup_patch_prolongation_operators(
    tests_u,jac_u,graddiv,qdegree;is_nonlinear=true
  )
  restrictions = setup_patch_restriction_operators(
    tests_u,prolongations,graddiv,qdegree;solver=IS_ConjugateGradientSolver(;reltol=1.e-6)
  )
  gmg = GMGLinearSolver(
    mh,trials_u,tests_u,biforms,
    prolongations,restrictions,
    pre_smoothers=smoothers,
    post_smoothers=smoothers,
    coarsest_solver=LUSolver(),
    maxiter=2,mode=:preconditioner,verbose=i_am_main(parts),is_nonlinear=true
  )

  solver_u = gmg
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  solver_u.log.depth = 3
  solver_p.log.depth = 3

  bblocks  = [NonlinearSystemBlock([1]) LinearSystemBlock();
              LinearSystemBlock()       BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)]
  coeffs = [1.0 1.0;
            0.0 1.0]
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts))
  solver.log.depth = 2

  nlsolver = NewtonSolver(solver;maxiter=20,atol=1e-14,rtol=1.e-7,verbose=i_am_main(parts))
  xh = solve(nlsolver,op)

  @test true
end

end # module
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

