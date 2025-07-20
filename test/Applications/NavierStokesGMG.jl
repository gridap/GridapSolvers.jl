# # Incompressible Navier-Stokes equations in a 2D/3D cavity, using GMG.
# 
# This example solves the incompressible Stokes equations, given by 
# 
# ```math
# \begin{align*}
# -\Delta u + \text{R}_e (u \nabla) u - \nabla p &= f \quad \text{in} \quad \Omega, \\
# \nabla \cdot u &= 0 \quad \text{in} \quad \Omega, \\
# u &= \hat{x} \quad \text{in} \quad \Gamma_\text{top} \subset \partial \Omega, \\
# u &= 0 \quad \text{in} \quad \partial \Omega \backslash \Gamma_\text{top} \\
# \end{align*}
# ```
# 
# where $\Omega = [0,1]^d$. 
# 
# We use a mixed finite-element scheme, with $Q_k \times P_{k-1}^{-}$ elements for the velocity-pressure pair. 
# 
# To solve the linear system, we use a FGMRES solver preconditioned by a block-triangular 
# Shur-complement-based preconditioner. We use an Augmented Lagrangian approach to 
# get a better approximation of the Schur complement. Details for this preconditoner can be 
# found in [Benzi and Olshanskii (2006)](https://epubs.siam.org/doi/10.1137/050646421).
# 
# The velocity block is solved using a Geometric Multigrid (GMG) solver. Due to the kernel 
# introduced by the Augmented-Lagrangian operator, we require special smoothers and prolongation/restriction
# operators. See [Schoberl (1999)](https://link.springer.com/article/10.1007/s002110050465) for more details.
module NavierStokesGMGApplication

using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: NonlinearSystemBlock, LinearSystemBlock, BiformBlock, BlockTriangularSolver

function get_patch_smoothers(sh,jac,qdegree)
  nlevs = num_levels(sh)
  smoothers = map(view(sh,1:nlevs-1)) do shl
    model = get_model(shl)
    ptopo = Geometry.PatchTopology(ReferenceFE{0},model)
    space = get_fe_space(shl)
    Ω  = Geometry.PatchTriangulation(model,ptopo)
    dΩ = Measure(Ω,qdegree)
    ap = (u,du,dv) -> jac(u,du,dv,dΩ)
    solver = PatchBasedSmoothers.PatchSolver(
      ptopo, space, space, ap;
      assembly = :star,
      collect_factorizations = true,
      is_nonlinear = true
    )
    return RichardsonSmoother(solver,10,0.2)
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
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[22])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26])
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

  u_walls = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
  u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

  tests_u  = TestFESpace(mh,reffe_u,dirichlet_tags=["walls","top"]);
  trials_u = TrialFESpace(tests_u,[u_walls,u_top]);
  U, V = get_fe_space(trials_u,1), get_fe_space(tests_u,1)
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  Re = 10.0
  ν = 1/Re
  α = 1.e2
  f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
  
  Π_Qh = LocalProjectionMap(divergence,reffe_p,qdegree)
  graddiv(u,v,dΩ) = ∫(α*(∇⋅v)⋅Π_Qh(u))dΩ

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
  smoothers = get_patch_smoothers(
    trials_u,jac_u,qdegree
  )
  prolongations = setup_patch_prolongation_operators(
    tests_u,jac_u,jac_u,qdegree;is_nonlinear=true,collect_factorizations=true
  )
  restrictions = setup_restriction_operators(
    tests_u,qdegree;mode=:residual,solver=CGSolver(JacobiLinearSolver())
  )
  gmg = GMGLinearSolver(
    trials_u,tests_u,biforms,
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
  solver = FGMRESSolver(20,P;atol=1e-11,rtol=1.e-8,verbose=i_am_main(parts))
  solver.log.depth = 2

  nlsolver = NewtonSolver(solver;maxiter=20,atol=1e-10,rtol=1.e-12,verbose=i_am_main(parts))
  xh = solve(nlsolver,op)

  @test true
end

end # module