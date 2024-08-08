# # Incompressible Navier-Stokes equations in a 2D/3D cavity
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
# The velocity block is solved directly using an exact solver.
module NavierStokesApplication

using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: LinearSystemBlock, NonlinearSystemBlock, BiformBlock, BlockTriangularSolver

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[22])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26])
end

function main(distribute,np,nc)
  parts = distribute(LinearIndices((prod(np),)))

  Dc = length(nc)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  model = CartesianDiscreteModel(parts,np,domain,nc)
  add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
  add_labels!(get_face_labeling(model))

  order = 2
  qdegree = 2*(order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  u_walls = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
  u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

  V = TestFESpace(model,reffe_u,dirichlet_tags=["walls","top"]);
  U = TrialFESpace(V,[u_walls,u_top]);
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  Re = 10.0
  ν = 1/Re
  α = 1.e2
  f = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)

  Π_Qh = LocalProjectionMap(divergence,Q)
  graddiv(u,v,dΩ) = ∫(α*Π_Qh(u,dΩ)⋅Π_Qh(v,dΩ))dΩ

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

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  jac_h(x,dx,dy) = jac(x,dx,dy,dΩ)
  res_h(x,dy) = res(x,dy,dΩ)
  op = FEOperator(res_h,jac_h,X,Y)

  solver_u = LUSolver()
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  solver_p.log.depth = 4

  bblocks  = [NonlinearSystemBlock() LinearSystemBlock();
              LinearSystemBlock()    BiformBlock((p,q) -> ∫(-(1.0/α)*p*q)dΩ,Q,Q)]
  coeffs = [1.0 1.0;
            0.0 1.0]  
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts))
  solver.log.depth = 2
  
  nlsolver = NewtonSolver(solver;maxiter=20,atol=1e-14,rtol=1.e-7,verbose=i_am_main(parts))
  xh = solve(nlsolver,op);

  @test true
end

end # module