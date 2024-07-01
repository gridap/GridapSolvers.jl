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
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"bottom",[1,2,5])
  add_tag_from_tags!(labels,"walls",[7,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
  add_tag_from_tags!(labels,"bottom",[1,2,3,4,9,10,13,14,21])
  add_tag_from_tags!(labels,"walls",[17,18,23,25,26])
end

function main(distribute,np,nc)
  parts = distribute(LinearIndices((prod(np),)))

  # Geometry
  Dc = length(nc)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  model = CartesianDiscreteModel(parts,np,domain,nc)
  add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
  add_labels!(get_face_labeling(model))

  # FE spaces
  order = 2
  qdegree = 2*(order+1)
  reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  u_bottom = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
  u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

  V = TestFESpace(model,reffe_u,dirichlet_tags=["bottom","top"]);
  U = TrialFESpace(V,[u_bottom,u_top]);
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

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

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  jac_h(x,dx,dy) = jac(x,dx,dy,dΩ)
  res_h(x,dy) = res(x,dy,dΩ)
  op = FEOperator(res_h,jac_h,X,Y)

  # Solver
  solver_u = LUSolver() # or mumps
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  solver_p.log.depth = 4

  diag_blocks  = [NonlinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
  bblocks = map(CartesianIndices((2,2))) do I
    (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
  end
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